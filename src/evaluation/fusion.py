#!/usr/bin/env python3
"""T36: soft voting between the full-image model and the crop model of the same family.

For each family the class probabilities of the full-image model (S01-S15, one row per photograph) and
of the per-crop model (S25-S32, one row per crop) are averaged with weight ``w`` on the full image:

- **photo level**: a photograph's full-image probabilities and the mean of its crops' probabilities,
  ``w * full + (1 - w) * mean(crops)``; one row per photograph, as the full-image files;
- **crop level**: a crop's probabilities and its photograph's full-image probabilities,
  ``w * full + (1 - w) * crop``; one row per crop, as the per-crop files.

Two rules: ``equal`` (w = 0.5, fixed before any result) and ``tuned`` (w from ``--grid`` with the best
validation macro-F1, per family and level; ties go to the weight nearest 0.5). The test split is never
used to choose. Validation and test files are both written, in the layout of the runs
(``<rule>/main/full/<family>/`` and ``<rule>/roi_crop/crop/<family>/``), so ``patient_rescore.py`` and
``contrast_compare.py`` score them against the original runs unchanged. Also writes
``fusion_weights.csv`` (validation macro-F1 at every weight). Output outside the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.evaluation.evaluate import FAMILY_ORDER  # noqa: E402
from src.evaluation.metrics import summary_metrics  # noqa: E402
from src.evaluation.single_box import photo_stem  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
FULL, CROP = ("main", "full"), ("roi_crop", "crop")
SPLITS = ("validation", "test")
GRID = [round(w, 1) for w in np.linspace(0, 1, 11)]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft voting between full-image and crop models.")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Experiments folder with main/full/<family>/ and roi_crop/crop/<family>/.")
    parser.add_argument("--output-root", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--families", nargs="+", default=list(FAMILY_ORDER))
    parser.add_argument("--grid", nargs="+", type=float, default=GRID, help="Weights on the full image to try.")
    return parser.parse_args(argv)


def prob_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c.startswith("prob_")]


def class_name(column: str) -> str:
    return column[len("prob_"):].replace("_", " ")


def photo_keys(full: pd.DataFrame, crops: pd.DataFrame, boxes: pd.DataFrame) -> Tuple[List, List]:
    """(group, file stem) of every full-image row and of every crop's photograph (as patient_rescore.py)."""
    full_keys = list(zip(full["pseudonimized_id"], full["relative_path"].map(photo_stem)))
    if len(set(full_keys)) != len(full_keys):
        raise ValueError("Two full-image rows share a group and file stem")
    missing = sorted(set(crops["relative_path"]) - set(boxes.index))
    if missing:
        raise ValueError(f"{len(missing)} crops are not in the box manifest: {missing[:3]}")
    box = boxes.loc[crops["relative_path"]]
    crop_keys = list(zip(box["pseudonimized_id"], box["filename"].map(photo_stem)))
    return full_keys, crop_keys


def fuse(full: pd.DataFrame, crops: pd.DataFrame, boxes: pd.DataFrame, weight: float) -> Dict[str, pd.DataFrame]:
    """Photo-level and crop-level fused prediction frames for one family and split."""
    columns = prob_columns(full)
    if columns != prob_columns(crops):
        raise ValueError("Full-image and crop files have different class columns")
    full_keys, crop_keys = photo_keys(full, crops, boxes)
    full_probs = pd.DataFrame(full[columns].to_numpy(dtype=float), index=pd.MultiIndex.from_tuples(full_keys))
    crop_probs = pd.DataFrame(crops[columns].to_numpy(dtype=float), index=pd.MultiIndex.from_tuples(crop_keys))

    # Every photograph needs its crops and every crop its photograph
    no_crops = sorted(set(full_keys) - set(crop_keys))
    no_photo = sorted(set(crop_keys) - set(full_keys))
    if no_crops or no_photo:
        raise ValueError(f"{len(no_crops)} photographs without crops, {len(no_photo)} crops without a photograph: "
                         f"{(no_crops + no_photo)[:3]}")

    crop_mean = crop_probs.groupby(level=[0, 1]).mean().loc[full_probs.index]
    photo = weight * full_probs.to_numpy() + (1 - weight) * crop_mean.to_numpy()
    crop = weight * full_probs.loc[crop_probs.index].to_numpy() + (1 - weight) * crop_probs.to_numpy()

    names = [class_name(c) for c in columns]
    frames = {}
    for level, base, probs in (("full", full, photo), ("crop", crops, crop)):
        frame = base.drop(columns=columns).copy()
        frame["predicted_class"] = [names[k] for k in probs.argmax(axis=1)]
        for k, column in enumerate(columns):
            frame[column] = probs[:, k]
        frames[level] = frame
    return frames


def macro_f1(frame: pd.DataFrame) -> float:
    columns = prob_columns(frame)
    names = [class_name(c) for c in columns]
    index = {n: k for k, n in enumerate(names)}
    probs = frame[columns].to_numpy(dtype=float)
    y_true = frame["true_class"].map(index).to_numpy()
    return summary_metrics(y_true, probs.argmax(axis=1), probs)["macro_f1"]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_root.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-root {out} is inside the repository; per-image files stay outside it")
    boxes = pd.read_csv(args.box_manifest, dtype=str, keep_default_na=False).set_index("roi_relative_path")
    grid = sorted(set(args.grid))

    weights = []
    for family in args.families:
        files = {split: {name: pd.read_csv(args.source_root / run / path / family / f"{split}_predictions.csv",
                                           dtype={"pseudonimized_id": str})
                         for name, (run, path) in (("full", FULL), ("crop", CROP))}
                 for split in SPLITS}
        # Validation macro-F1 at every weight, per level; the tuned weight is the best (ties: nearest 0.5)
        scores = {w: {level: macro_f1(frame) for level, frame in
                      fuse(files["validation"]["full"], files["validation"]["crop"], boxes, w).items()} for w in grid}
        chosen = {"equal": {"full": 0.5, "crop": 0.5}}
        chosen["tuned"] = {level: max(grid, key=lambda w: (round(scores[w][level], 10), -abs(w - 0.5)))
                           for level in ("full", "crop")}
        for w in grid:
            for level in ("full", "crop"):
                weights.append({"family": family, "level": "photo" if level == "full" else "crop", "weight_full": w,
                                "validation_macro_f1": scores[w][level],
                                "tuned": w == chosen["tuned"][level], "equal": w == 0.5})
        for rule, by_level in chosen.items():
            for split in SPLITS:
                for level, (run, path) in (("full", FULL), ("crop", CROP)):
                    frame = fuse(files[split]["full"], files[split]["crop"], boxes, by_level[level])[level]
                    target = out / rule / run / path / family
                    target.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(target / f"{split}_predictions.csv", index=False)
        print(f"{family}: tuned weight on the full image, photo level {chosen['tuned']['full']}, "
              f"crop level {chosen['tuned']['crop']}", flush=True)

    pd.DataFrame(weights).to_csv(out / "fusion_weights.csv", index=False)
    print(f"Wrote fused predictions (equal, tuned) and fusion_weights.csv to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
