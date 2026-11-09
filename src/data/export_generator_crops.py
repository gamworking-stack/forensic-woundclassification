#!/usr/bin/env python3
"""T34: the training wound crops as a dataset for the image generators (StyleGAN2-ADA, diffusion).

Only training-split crops are written (with ``--cv-folds``/``--fold``: the training folds of that
CV fold), so no validation or test crop ever reaches a generator. Every crop is padded to a square
and resized exactly as the classifiers see it (``PadToSquare``, then ``--size`` px), so generated
crops come in the classifiers' own form. All 7 classes are written: the generators are
class-conditional and learn skin, light and texture from every class.

Writes ``<class index>/<n>.png``, ``dataset.json`` (StyleGAN's ``{"labels": [[file, class], ...]}``),
``crops.csv`` (file, class, source crop, case; for the memorisation check) and ``dataset_info.json``
(classes, counts, settings). Patient images: run on Raptor, outputs stay there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.training.data import GROUP_COLUMN, load_manifest, prepare_rows  # noqa: E402
from src.training.transforms import PadToSquare  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training wound crops as a class-labelled generator dataset.")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--image-root", type=Path, required=True, help="Per-box crop folder (roi_box_p008).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--cv-folds", type=Path, default=None, help="CV fold table (T12b): export that fold's training folds.")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args(argv)
    if (args.cv_folds is None) != (args.fold is None):
        parser.error("--cv-folds and --fold go together")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; patient-derived images stay outside it")

    # Training crops only, with the classifiers' class order (sorted names)
    df = load_manifest(args.manifest, "crop", cv_folds=args.cv_folds, fold=args.fold)
    df, classes = prepare_rows(df, "crop", args.image_root)
    train = df[df["split"] == "train"].reset_index(drop=True)

    pad = PadToSquare()
    labels, rows = [], []
    for i, row in train.iterrows():
        name = f"{int(row['label_index'])}/{i:05d}.png"
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        with Image.open(row["image_path"]) as image:
            square = pad(image.convert("RGB")).resize((args.size, args.size), Image.LANCZOS)
        square.save(out / name)
        labels.append([name, int(row["label_index"])])
        rows.append({"file": name, "class": classes[int(row["label_index"])],
                     "roi_relative_path": row["roi_relative_path"], "case": row[GROUP_COLUMN]})

    (out / "dataset.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    crops = pd.DataFrame(rows)
    crops.to_csv(out / "crops.csv", index=False)
    info = {"classes": classes, "size": args.size, "manifest": str(args.manifest), "image_root": str(args.image_root),
            "cv_folds": str(args.cv_folds) if args.cv_folds else None, "fold": args.fold,
            "crops_per_class": {c: int((crops["class"] == c).sum()) for c in classes},
            "cases": int(crops["case"].nunique())}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Wrote {len(crops)} training crops ({args.size} px) to {out}: {info['crops_per_class']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
