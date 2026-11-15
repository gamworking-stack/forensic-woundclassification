#!/usr/bin/env python3
"""Detector arm 4Y (T37): YOLO11 fine-tuned on the training split's manual boxes.

The arm follows T19's 4F protocol with a YOLO detector in place of Faster R-CNN:
- one class, ``wound`` (class-agnostic, as every T19 arm);
- three configurations, YOLO11 s / m / l from COCO weights (``yolo11<size>``);
- the size and the operating threshold are chosen on the **validation** photos by ``select.py --arm 4Y``
  (AP@0.5, ties to the smaller model; F1 threshold; recall gate >= 0.50);
- test photos are detected once, with the chosen model, after ``selection.json`` exists.

Subcommands:
- ``prepare``: the YOLO dataset of the train and validation photos (never test). Each photo is
  EXIF-oriented, shrunk so its longest side is ``STORE_SIDE`` and saved as JPEG; its unpadded manual
  boxes (``reference_boxes.csv``) become one label file in YOLO's normalised ``class cx cy w h`` form.
  ``photos.csv`` maps every image file to its photo; ``data.yaml`` points YOLO at both splits.
- ``train``: one YOLO11 size with the frozen recipe (``TRAIN_SETTINGS``). YOLO keeps the epoch with
  the best validation fitness as ``weights/best.pt``.
- ``detect``: ``best.pt`` on the photos of given splits, shrunk exactly as for training, with
  confidence ``CONFIDENCE``; boxes are scaled back to source pixels and written as a T19 detections
  file. Test only with the frozen ``selection.json`` that chose this run (``test_allowed``).

``ultralytics`` is imported only inside ``train`` and ``detect``; it lives in the Raptor venv
``venv_exp4_yolo`` (requirements-exp4.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.boxes import COORDINATES, as_boxes  # noqa: E402
from src.detection.common import PHOTO, detection_rows, load_photo, photo_table, test_allowed, write_detections  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SIZES = ("s", "m", "l")  # simplest first: select.py gives ties to the earlier configuration
STORE_SIDE = 1600        # longest side of the photos YOLO trains and detects on
YOLO_FOLDERS = {"train": "train", "validation": "val"}
TRAIN_SETTINGS = {
    "imgsz": 1280, "epochs": 150, "patience": 30, "batch": 16, "seed": 42, "deterministic": True,
    "single_cls": True, "optimizer": "auto", "flipud": 0.5, "fliplr": 0.5, "plots": False,
}
CONFIDENCE = 0.001
YOLO_NMS_IOU = 0.7
MAX_DETECTIONS = 300

Predict = Callable[[object], Tuple[np.ndarray, np.ndarray]]


def config_name(size: str) -> str:
    return f"yolo11{size}"


def config_order() -> List[str]:
    return [config_name(s) for s in SIZES]


def outside_repo(path: Path) -> Path:
    path = path.resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"{path} is inside the repository; photos and detections stay outside it")
    return path


def shrink(image, side: int = STORE_SIDE):
    """The photo with its longest side at most ``side`` (bicubic), and the scale from source pixels."""
    scale = min(1.0, side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), resample=3)
    return image, scale


def label_lines(boxes: np.ndarray, width: float, height: float) -> List[str]:
    """YOLO label rows (class 0, centre and size as fractions of the image) for source-pixel boxes."""
    boxes = as_boxes(boxes)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    w = (boxes[:, 2] - boxes[:, 0]) / width
    h = (boxes[:, 3] - boxes[:, 1]) / height
    values = np.clip(np.stack([cx, cy, w, h], axis=1), 0.0, 1.0)
    return [f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}" for a, b, c, d in values]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- prepare ----------

def prepare(args: argparse.Namespace) -> int:
    out = outside_repo(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; the YOLO dataset is written once")
    # Train and validation photos only: test never enters the YOLO dataset
    photos = photo_table(args.box_manifest, tuple(YOLO_FOLDERS))
    reference = pd.read_csv(args.reference, dtype={PHOTO: str})
    boxes_of = {p: g[COORDINATES].to_numpy(float) for p, g in reference.groupby(PHOTO)}
    missing = sorted(set(photos[PHOTO]) - set(boxes_of))
    if missing:
        raise SystemExit(f"{len(missing)} photos have no reference box, e.g. {missing[:3]}")

    rows = []
    for index, row in enumerate(photos.to_dict("records"), start=1):
        # The upright photo, shrunk, saved under a neutral name; its boxes as a YOLO label file
        folder = YOLO_FOLDERS[row["split"]]
        name = f"{folder}_{index:04d}"
        image, scale = shrink(load_photo(args.image_root, row))
        for kind in ("images", "labels"):
            (out / kind / folder).mkdir(parents=True, exist_ok=True)
        image.save(out / "images" / folder / f"{name}.jpg", format="JPEG", quality=95)
        lines = label_lines(boxes_of[row[PHOTO]], row["source_width"], row["source_height"])
        (out / "labels" / folder / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append({"image": f"images/{folder}/{name}.jpg", PHOTO: row[PHOTO], "split": row["split"],
                     "boxes": len(lines), "scale": scale, "source_width": row["source_width"],
                     "source_height": row["source_height"]})

    pd.DataFrame(rows).to_csv(out / "photos.csv", index=False)
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: wound\n", encoding="utf-8")
    counts = pd.DataFrame(rows).groupby("split").agg(photos=(PHOTO, "size"), boxes=("boxes", "sum"))
    print(f"YOLO dataset written to {out}:\n{counts.to_string()}")
    return 0


# ---------- train ----------

def train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    run = outside_repo(args.output_dir)
    if run.exists():
        raise SystemExit(f"{run} exists; a training run is written once")
    settings = {**TRAIN_SETTINGS, **({"epochs": args.epochs} if args.epochs else {})}
    model = YOLO(str(args.weights))
    model.train(data=str(args.data_dir / "data.yaml"), project=str(run.parent), name=run.name, exist_ok=False,
                workers=args.workers, device=args.device, **settings)
    # What was trained from what, for the ticket and the verification
    record = {"config": run.name, "weights": str(args.weights), "weights_sha256": sha256(args.weights),
              "data": str(args.data_dir), "settings": settings,
              "best_sha256": sha256(run / "weights" / "best.pt")}
    (run / "run_config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Trained {run.name}: {run / 'weights' / 'best.pt'}")
    return 0


# ---------- detect ----------

def detect_photos(predict: Predict, photos: pd.DataFrame, image_root: Path, side: int = STORE_SIDE) -> List[Dict]:
    """Raw detections in source pixels: each photo shrunk as for training, boxes scaled back."""
    rows: List[Dict] = []
    for row in photos.to_dict("records"):
        image, scale = shrink(load_photo(image_root, row), side)
        boxes, scores = predict(image)
        rows += detection_rows(row[PHOTO], np.asarray(boxes, dtype=float).reshape(-1, 4) / scale,
                               np.asarray(scores, dtype=float).reshape(-1), row["source_width"], row["source_height"])
    return rows


def yolo_predictor(weights: Path, device: str) -> Predict:
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(image) -> Tuple[np.ndarray, np.ndarray]:
        # A PIL image is RGB; ultralytics converts it to its own BGR order
        result = model.predict(image, imgsz=TRAIN_SETTINGS["imgsz"], conf=CONFIDENCE, iou=YOLO_NMS_IOU,
                               max_det=MAX_DETECTIONS, device=device, verbose=False)[0]
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

    return predict


def detect(args: argparse.Namespace) -> int:
    # --detector-dir: the run that the frozen selection chose, written as <output-dir>/<config>.csv
    if args.detector_dir is not None:
        if args.selection is None or args.output_dir is None:
            raise SystemExit("--detector-dir needs --selection and --output-dir")
        chosen = json.loads(args.selection.read_text(encoding="utf-8"))["chosen_config"]
        args.run_dir, args.output = args.detector_dir / chosen, args.output_dir / f"{chosen}.csv"
    if args.run_dir is None or args.output is None:
        raise SystemExit("Give --run-dir and --output, or --detector-dir, --selection and --output-dir")
    if "test" in args.splits:
        test_allowed(args.selection, args.run_dir.name)
    outside_repo(args.output)
    photos = photo_table(args.box_manifest, args.splits)
    predict = yolo_predictor(args.run_dir / "weights" / "best.pt", args.device)
    write_detections(detect_photos(predict, photos, args.image_root), args.output)
    print(f"Wrote detections of {len(photos)} photos ({'+'.join(args.splits)}) to {args.output}")
    return 0


# ---------- script ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector arm 4Y: YOLO11 fine-tuned on the manual boxes (T37).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="The YOLO dataset, outside the repository.")
    p.add_argument("--reference", type=Path, default=DATA_DIR / "roi_auto/reference_boxes.csv")
    t = sub.add_parser("train")
    t.add_argument("--size", choices=SIZES, required=True)
    t.add_argument("--weights", type=Path, required=True, help="COCO weights yolo11<size>.pt")
    t.add_argument("--data-dir", type=Path, required=True, help="The prepared YOLO dataset.")
    t.add_argument("--output-dir", type=Path, required=True, help="<detector folder>/yolo11<size>")
    t.add_argument("--epochs", type=int, default=None, help="Smoke runs only.")
    t.add_argument("--workers", type=int, default=8)
    d = sub.add_parser("detect")
    d.add_argument("--run-dir", type=Path, default=None, help="A training run, named yolo11<size>.")
    d.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), required=True)
    d.add_argument("--output", type=Path, default=None)
    d.add_argument("--detector-dir", type=Path, default=None,
                   help="Instead of --run-dir: the folder of all runs; the selection's chosen run is used.")
    d.add_argument("--output-dir", type=Path, default=None, help="With --detector-dir: writes <config>.csv here.")
    d.add_argument("--selection", type=Path, default=None)
    d.add_argument("--image-root", type=Path, required=True)
    for s in (p, t, d):
        s.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    for s in (t, d):
        s.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    if args.command == "train" and args.output_dir.name != config_name(args.size):
        parser.error(f"--output-dir must be named {config_name(args.size)} (the configuration name)")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return {"prepare": prepare, "train": train, "detect": detect}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
