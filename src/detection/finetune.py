#!/usr/bin/env python3
"""Detector arm 4F: Faster R-CNN fine-tuned on the training split's manual boxes (T19 §2.2, §2.5).

``train``: one run of the frozen recipe at one learning rate.
- Model: ``fasterrcnn_resnet50_fpn_v2`` from COCO weights, 2 classes (background, wound),
  ``min_size=1024``, ``max_size=1365``.
- Augmentation: horizontal and vertical flips, mild brightness/contrast jitter.
- Optimisation: SGD (momentum 0.9, weight decay 1e-4), batch 4, linear warmup over 500 iterations,
  at most 60 epochs, patience 10 on class-agnostic AP@0.5 of the validation photos; fp32, seed 42.
- Writes ``best.pt``, ``history.csv`` and ``validation.csv`` (the validation detections of the
  best epoch).

Cross-fitting:
- ``train --holdout-fold k --epochs E --lr X`` learns from the training cases outside fold ``k``
  of ``cv_folds.csv``, for exactly ``E`` epochs (the main run's best epoch count, no early
  stopping).
- It then writes ``train_fold<k>.csv`` with the detections of fold ``k``'s training photos only.

``detect``: the main run's ``best.pt`` on given splits; test only with the frozen ``selection.json``.

Photos are cached with their longest side at ``max_size`` (what the model's own resize would
produce); boxes are scaled in and detections scaled back to source pixels. The 3 learning-rate
runs are the arm's configurations, named ``lr<value>``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.boxes import COORDINATES  # noqa: E402
from src.detection.common import (  # noqa: E402
    PHOTO,
    detection_rows,
    load_photo,
    photo_table,
    test_allowed,
    write_detections,
)
from src.detection.evaluate_detection import candidates, coco_ap, coco_images, reference_photos  # noqa: E402
from src.training.data import GROUP_COLUMN  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402


LR_GRID = (0.005, 0.01, 0.02)
MIN_SIZE, MAX_SIZE = 1024, 1365
BATCH_SIZE = 4
WARMUP_ITERATIONS = 500
MAX_EPOCHS = 60
PATIENCE = 10
MOMENTUM, WEIGHT_DECAY = 0.9, 1e-4
SCORE_FLOOR, DETECTIONS_PER_IMAGE = 0.001, 300
SEED = 42


def config_order() -> List[str]:
    return [f"lr{lr:g}" for lr in LR_GRID]


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained: bool = True):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Faster R-CNN from COCO weights, our input size, and raw boxes kept down to a very low score
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None, min_size=MIN_SIZE, max_size=MAX_SIZE,
                                       box_score_thresh=SCORE_FLOOR, box_detections_per_img=DETECTIONS_PER_IMAGE)
    # Replace COCO's 91-class head with 2 classes: background and wound
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    return model


class PhotoCache:
    """Photos shrunk so the longest side is ``max_side``, with each one's scale to source pixels."""

    def __init__(self, photos: pd.DataFrame, image_root: Path, max_side: Optional[int] = None) -> None:
        max_side = max_side or MAX_SIZE
        self.images, self.scales = {}, {}
        # Open each photo once, shrink it so the longest side is max_side, remember the scale
        for row in photos.to_dict("records"):
            image = load_photo(image_root, row)
            scale = min(1.0, max_side / max(image.size))
            if scale < 1.0:
                image = image.resize((round(image.width * scale), round(image.height * scale)), resample=3)
            self.images[row[PHOTO]], self.scales[row[PHOTO]] = image, scale


def augment(image, boxes: np.ndarray, rng: random.Random):
    """Horizontal and vertical flips (p = 0.5 each) and mild brightness/contrast jitter (±20%)."""
    from PIL import ImageEnhance

    width, height = image.size
    # Flip left-right and/or top-bottom (half the time each), moving the boxes with the image
    boxes = boxes.copy()
    if rng.random() < 0.5:
        image = image.transpose(0)  # FLIP_LEFT_RIGHT
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    if rng.random() < 0.5:
        image = image.transpose(1)  # FLIP_TOP_BOTTOM
        boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
    # Mild brightness and contrast change (plus or minus 20%)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    return image, boxes


def to_tensor(image):
    import torch
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)


def detect_photos(model, cache: PhotoCache, photos: Sequence[str], device) -> List[Dict]:
    """Raw detections in source pixels for ``photos``."""
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for photo in photos:
            # Detect on the shrunk photo, then scale the boxes back to source pixels
            image, scale = cache.images[photo], cache.scales[photo]
            out = model([to_tensor(image).to(device)])[0]
            boxes = out["boxes"].float().cpu().numpy() / scale
            scores = out["scores"].float().cpu().numpy()
            width, height = round(image.width / scale), round(image.height / scale)
            rows += detection_rows(photo, boxes, scores, width, height)
    return rows


def validation_ap50(rows: List[Dict], val_photos: pd.DataFrame, max_share: float) -> float:
    # Validation AP@0.5 with the same step-1 rules as the selection
    detections = pd.DataFrame(rows, columns=[PHOTO, "score", *COORDINATES])
    return coco_ap(coco_images(val_photos, candidates(detections, val_photos, max_share)))["ap50"]


def train(args: argparse.Namespace) -> int:
    import torch

    # Seed everything and prepare the output folder, reference boxes and limits
    seed_all(SEED)
    device = torch.device(args.device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference, dtype={PHOTO: str, GROUP_COLUMN: str})
    limits = json.loads(args.limits.read_text(encoding="utf-8"))
    # Training photos; for cross-fitting, hold one fold's cases out
    photos = photo_table(args.box_manifest, ("train", "validation"))
    train_photos = photos[photos["split"] == "train"]
    if args.holdout_fold is not None:  # cross-fitting: learn without fold k's training cases
        folds = pd.read_csv(args.cv_folds, dtype=str).set_index(GROUP_COLUMN)["fold"].astype(int)
        fold_of = train_photos[GROUP_COLUMN].map(folds)
        if fold_of.isna().any():
            raise ValueError(f"{int(fold_of.isna().sum())} training photos have no fold in {args.cv_folds}")
        holdout = train_photos[fold_of == args.holdout_fold]
        train_photos = train_photos[fold_of != args.holdout_fold]
        if holdout.empty:
            raise ValueError(f"Fold {args.holdout_fold} has no training photos")
    # Manual boxes of the training photos, validation photos for early stopping, and the photo cache
    truth = reference_photos(reference, ("train",))["truth"]
    val_photos = reference_photos(reference, ("validation",))
    cache = PhotoCache(photos if args.holdout_fold is None else pd.concat([train_photos, holdout]), args.image_root)

    # Model, SGD optimiser and a linear learning-rate warmup
    model = build_model(not args.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: min(1.0, (it + 1) / WARMUP_ITERATIONS))
    rng = random.Random(SEED)
    names = list(train_photos[PHOTO])
    epochs = args.epochs if args.epochs is not None else args.max_epochs
    history, best_ap, best_epoch, since_best, started = [], -1.0, 0, 0, time.time()
    # Write the run's settings before training starts
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "min_size": MIN_SIZE, "max_size": MAX_SIZE, "batch_size": BATCH_SIZE, "warmup_iterations": WARMUP_ITERATIONS,
              "patience": None if args.epochs is not None else PATIENCE, "seed": SEED,
              "train_photos": len(names), "train_boxes": int(sum(len(truth[n]) for n in names))}
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        # One epoch: shuffle the photos and train in batches of 4
        model.train()
        rng.shuffle(names)
        losses = []
        for start in range(0, len(names), BATCH_SIZE):
            images, targets = [], []
            for name in names[start:start + BATCH_SIZE]:
                # Scale the manual boxes to the shrunk photo, augment, and build the model's targets
                scale = cache.scales[name]
                image, boxes = augment(cache.images[name], truth[name] * scale, rng)
                images.append(to_tensor(image).to(device))
                targets.append({"boxes": torch.as_tensor(boxes, dtype=torch.float32, device=device),
                                "labels": torch.ones(len(boxes), dtype=torch.int64, device=device)})
            # Faster R-CNN returns its losses; sum them and take one SGD step
            loss = sum(model(images, targets).values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            warmup.step()
            losses.append(float(loss))
        # Main runs: score validation each epoch, keep the best checkpoint, count epochs without improvement
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "elapsed_minutes": (time.time() - started) / 60}
        if args.epochs is None:  # main runs: early stopping on validation AP@0.5
            row["validation_ap50"] = validation_ap50(detect_photos(model, cache, list(val_photos.index), device),
                                                     val_photos, limits["max_box_share"])
            if row["validation_ap50"] > best_ap:
                best_ap, best_epoch, since_best = row["validation_ap50"], epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation_ap50": best_ap}, out / "best.pt")
            else:
                since_best += 1
        # Log the epoch; stop after PATIENCE epochs without improvement
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if args.epochs is None and since_best >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best validation AP@0.5 {best_ap:.4f} at epoch {best_epoch}", flush=True)
            break

    if args.epochs is not None:  # cross-fitted run: the last epoch boxes its held-out training photos
        torch.save({"model": model.state_dict(), "epoch": epochs}, out / "last.pt")
        rows = detect_photos(model, cache, list(holdout[PHOTO]), device)
        write_detections(rows, out / f"train_fold{args.holdout_fold}.csv")
        print(f"Fold {args.holdout_fold}: {len(holdout)} held-out training photos boxed -> {out}")
        return 0
    # Main run: reload the best epoch and write its validation detections (the selection uses them)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
    write_detections(detect_photos(model, cache, list(val_photos.index), device), out / "validation.csv")
    print(f"Best epoch {best_epoch}, validation AP@0.5 {best_ap:.4f}; wrote {out / 'validation.csv'}")
    return 0


def detect(args: argparse.Namespace) -> int:
    import torch

    # Test needs a passing selection.json that chose this run
    if "test" in args.splits:
        test_allowed(args.selection, args.run_dir.name)
    # Load the run's best checkpoint and detect on the requested photos
    device = torch.device(args.device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.run_dir / "best.pt", map_location=device, weights_only=True)["model"])
    photos = photo_table(args.box_manifest, args.splits)
    cache = PhotoCache(photos, args.image_root)
    write_detections(detect_photos(model, cache, list(photos[PHOTO]), device), args.output)
    print(f"Wrote detections of {len(photos)} photos ({'+'.join(args.splits)}) to {args.output}")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector arm 4F: fine-tuned Faster R-CNN (T19).")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--output-dir", type=Path, required=True)
    t.add_argument("--image-root", type=Path, required=True)
    t.add_argument("--holdout-fold", type=int, default=None, help="Cross-fitting: leave this fold's training cases out.")
    t.add_argument("--epochs", type=int, default=None, help="Cross-fitting: fixed epoch count (main run's best).")
    t.add_argument("--cv-folds", type=Path, default=DATA_DIR / "splits/cv_folds.csv")
    t.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    t.add_argument("--no-pretrained", action="store_true", help="Tests only.")
    d = sub.add_parser("detect")
    d.add_argument("--run-dir", type=Path, required=True, help="A main run folder, named lr<value>.")
    d.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--selection", type=Path, default=None)
    d.add_argument("--image-root", type=Path, required=True)
    for p in (t, d):
        p.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
        p.add_argument("--reference", type=Path, default=DATA_DIR / "roi_auto/reference_boxes.csv")
        p.add_argument("--limits", type=Path, default=DATA_DIR / "roi_auto/reference_limits.json")
        p.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    # Cross-fitting needs both the held-out fold and the fixed epoch count
    if args.command == "train" and (args.holdout_fold is None) != (args.epochs is None):
        parser.error("--holdout-fold and --epochs go together (cross-fitting)")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return train(args) if args.command == "train" else detect(args)


if __name__ == "__main__":
    sys.exit(main())
