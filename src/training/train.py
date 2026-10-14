#!/usr/bin/env python3
"""Fine-tune one model family on one input path and write per-image prediction files.

Training keeps the checkpoint with the best validation macro-F1 (ties: lower validation loss)
and stops after ``patience`` epochs without improvement. The test split is loaded and predicted
only after training has ended, with that checkpoint; ``--skip-test`` (T08b tuning runs) leaves it
out entirely. Metrics are computed later from the prediction files (T08).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.common import (  # noqa: E402
    Logger,
    environment_info,
    git_state,
    prediction_frame,
    resolve_device,
)
from src.training.config import parse_args  # noqa: E402
from src.training.data import (  # noqa: E402
    GROUP_COLUMN,
    SPLIT_NAMES,
    ManifestImageDataset,
    load_manifest,
    prepare_rows,
)
from src.training.engine import (  # noqa: E402
    EarlyStopping,
    capture_rng,
    classification_metrics,
    load_checkpoint,
    predict,
    restore_rng,
    save_checkpoint,
    seed_everything,
    seed_worker,
    train_one_epoch,
)
from src.training.models import build_model, input_size  # noqa: E402
from src.training.photo import PHOTO_INPUT_PATHS, manifest_photos, photo_mean, photo_weights  # noqa: E402
from src.training.transforms import build_transforms  # noqa: E402


HISTORY_COLUMNS = [
    "model", "epoch", "learning_rate",
    "train_loss", "train_accuracy", "train_balanced_accuracy", "train_macro_f1",
    "validation_loss", "validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1",
    "elapsed_minutes",
]


def compute_class_weights(train_labels: np.ndarray, num_classes: int,
                          sample_weight: Optional[np.ndarray] = None) -> torch.Tensor:
    """Inverse class frequency on the train split, normalized to mean 1.

    With ``sample_weight`` (T17 photo weights) the frequency is the weighted count, i.e. photos.
    """
    counts = np.bincount(train_labels, weights=sample_weight, minlength=num_classes).astype(float)
    inverse = 1.0 / counts
    return torch.tensor(inverse / inverse.mean(), dtype=torch.float32)


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "train.log")

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    image_size = cfg.image_size or input_size(cfg.family)

    df = load_manifest(cfg.manifest, cfg.input_path, cfg.full_scope, cfg.roi_manifest, cfg.smoke,
                       cfg.cv_folds, cfg.fold)
    df, classes = prepare_rows(df, cfg.input_path, cfg.image_root)

    # T17 per-box crops: each photo counts once in the loss and selection is on photo level.
    per_box = cfg.input_path in PHOTO_INPUT_PATHS
    train_labels = df.loc[df["split"] == "train", "label_index"].to_numpy()
    train_weights = photo_weights(manifest_photos(df[df["split"] == "train"])) if per_box else None
    class_weights = compute_class_weights(train_labels, len(classes), train_weights)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if cfg.class_weights else None,
        label_smoothing=cfg.label_smoothing,
    )

    def make_loader(split: str, train: bool, generator: Optional[torch.Generator] = None
                    ) -> Tuple[pd.DataFrame, DataLoader]:
        rows = df[df["split"] == split]
        dataset = ManifestImageDataset(
            rows["image_path"].tolist(),
            rows["label_index"].tolist(),
            build_transforms(image_size, train),
            cache_side=2 * image_size if cfg.cache else None,
            threads=max(cfg.num_workers, 4),
            weights=photo_weights(manifest_photos(rows)) if per_box and train else None,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=train,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
            # A last batch of one image breaks BatchNorm in train mode.
            drop_last=train and len(dataset) % cfg.batch_size == 1,
        )
        return rows, loader

    run_config = {
        **cfg.to_json(),
        "image_size": image_size,
        "classes": classes,
        "class_weights": [round(float(w), 6) for w in class_weights],
        "images_per_split": {s: int((df["split"] == s).sum()) for s in SPLIT_NAMES},
        "cases_per_split": {s: int(df.loc[df["split"] == s, GROUP_COLUMN].nunique()) for s in SPLIT_NAMES},
        **({"photos_per_split": {s: len(set(manifest_photos(df[df["split"] == s]))) for s in SPLIT_NAMES},
            "selection_unit": "photo", "loss_weights": "1 / crops per photo"} if per_box else {}),
        **git_state(),
        "environment": environment_info(device),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (out / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    last_path, best_path = out / "last.pt", out / "best.pt"
    state = load_checkpoint(last_path, device) if cfg.resume and last_path.exists() else None
    history: List[Dict[str, Any]] = state["history"] if state else []
    stopper = EarlyStopping(**state["early_stopping"]) if state else EarlyStopping(cfg.patience)
    finished = stopper.stopped or len(history) >= cfg.max_epochs

    model = build_model(cfg.family, len(classes), cfg.pretrained).to(device)
    log(f"{cfg.family} | {cfg.input_path} | {device} | {image_size}px | "
        f"images {run_config['images_per_split']} | cases {run_config['cases_per_split']}")

    if finished:
        log(f"Training already finished at epoch {len(history)} (best epoch {stopper.best_epoch}).")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        start_epoch, elapsed_before = 1, 0.0
        if state:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
            generator.set_state(state["generator"])
            start_epoch = state["epoch"] + 1
            elapsed_before = history[-1]["elapsed_minutes"]
            log(f"Resuming from epoch {start_epoch}")

        _, train_loader = make_loader("train", True, generator)
        val_rows, val_loader = make_loader("validation", False)
        started = time.time()
        for epoch in range(start_epoch, cfg.max_epochs + 1):
            train_loss, train_y, train_pred = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_y, val_probs = predict(model, val_loader, criterion, device)
            train_m = classification_metrics(train_y, train_pred, len(classes))
            if per_box:  # selection metric on photo level: mean of each photo's crop probabilities
                _, val_y, val_probs = photo_mean(manifest_photos(val_rows), val_y, val_probs)
            val_m = classification_metrics(val_y, val_probs.argmax(axis=1), len(classes))
            row = {
                "model": cfg.family,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_m.items()},
                "validation_loss": val_loss,
                **{f"validation_{k}": v for k, v in val_m.items()},
                "elapsed_minutes": elapsed_before + (time.time() - started) / 60,
            }
            history.append(row)
            pd.DataFrame(history, columns=HISTORY_COLUMNS).to_csv(out / "training_history.csv", index=False)
            log(json.dumps(row))

            if stopper.step(epoch, val_m["macro_f1"], val_loss):
                save_checkpoint(best_path, {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_macro_f1": val_m["macro_f1"],
                    "validation_loss": val_loss,
                    "family": cfg.family,
                    "classes": classes,
                })
            save_checkpoint(last_path, {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "early_stopping": stopper.state_dict(),
                "history": history,
                "rng": capture_rng(),
                "generator": generator.get_state(),
            })
            if stopper.stopped:
                log(f"Early stopping {cfg.family} at epoch {epoch}; "
                    f"best validation macro-F1={stopper.best_f1:.6f} at epoch {stopper.best_epoch}")
                break
        del train_loader, val_loader  # shut down persistent workers before predicting

    prediction_splits = ("validation",) if cfg.skip_test else ("validation", "test")
    prediction_paths = {split: out / f"{split}_predictions.csv" for split in prediction_splits}
    if finished and all(p.exists() for p in prediction_paths.values()):
        log("Prediction files already present; nothing to do.")
        return 0

    chosen = load_checkpoint(best_path if cfg.checkpoint == "best" else last_path, device)
    model.load_state_dict(chosen["model"])
    # The test split is loaded here for the first time, after training has ended.
    for split, path in prediction_paths.items():
        rows, loader = make_loader(split, False)
        _, labels, probs = predict(model, loader, criterion, device)
        prediction_frame(rows, labels, probs, classes, photo=per_box).to_csv(path, index=False)
    detail = (f" (validation macro-F1={chosen['validation_macro_f1']:.6f})" if cfg.checkpoint == "best" else "")
    log(f"Wrote predictions from {cfg.checkpoint} epoch {chosen['epoch']}{detail} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
