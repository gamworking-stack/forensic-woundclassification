#!/usr/bin/env python3
"""T34: blinded real-or-generated rating of wound crops by the researcher.

``make``: for each class, ``--per-source`` real training crops and as many crops from each generator
(only those the memorisation check did not flag), all as 256 px PNG so size and format give nothing
away, shuffled and renamed ``rate_<n>.png``. Writes ``images/``, ``rating_sheet.csv`` (item, class,
an empty ``rating`` column to fill with real or generated) and, separately, ``answer_key.csv``
(item, source, original file), which stays on Raptor until the sheet is filled in.

``score``: the filled sheet against the key: per source and class, how often crops were called real.
A generator whose crops are called real as often as the real crops is indistinguishable to the rater.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.export_generator_crops import REPO_ROOT  # noqa: E402
from src.evaluation.generate_crops import SMALL_CLASSES  # noqa: E402

SIZE = 256


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blinded real-or-generated rating set, and its scoring.")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--samples", type=Path, nargs="+", required=True, help="Folders written by generate_crops.py.")
    make.add_argument("--nearest", type=Path, required=True, help="nearest.csv of generated_quality.py (flags).")
    make.add_argument("--train-dir", type=Path, required=True, help="Generator dataset with crops.csv (real crops).")
    make.add_argument("--classes", nargs="+", default=list(SMALL_CLASSES))
    make.add_argument("--per-source", type=int, default=10)
    make.add_argument("--output-dir", type=Path, required=True, help="Images and sheet (no key).")
    make.add_argument("--key-dir", type=Path, required=True, help="Where answer_key.csv goes; not shown to the rater.")
    make.add_argument("--seed", type=int, default=0)
    score = sub.add_parser("score")
    score.add_argument("--sheet", type=Path, required=True)
    score.add_argument("--key", type=Path, required=True)
    return parser.parse_args(argv)


def make(args: argparse.Namespace) -> int:
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; patient-derived images stay outside it")
    rng = np.random.default_rng(args.seed)
    flagged = set(pd.read_csv(args.nearest).query("flagged")["file"])
    real = pd.read_csv(args.train_dir / "crops.csv")

    picks = []
    for name in args.classes:
        files = real.loc[real["class"] == name, "file"].to_numpy()
        picks += [("real", name, args.train_dir / f) for f in rng.choice(files, args.per_source, replace=False)]
        for folder in args.samples:
            samples = pd.read_csv(folder / "samples.csv").query("`class` == @name")
            paths = [folder / f for f in samples["file"] if str(folder / f) not in flagged]
            source = samples["generator"].iloc[0]
            picks += [(source, name, paths[i]) for i in rng.choice(len(paths), args.per_source, replace=False)]

    order = rng.permutation(len(picks))
    (out / "images").mkdir(parents=True, exist_ok=True)
    sheet, key = [], []
    for n, i in enumerate(order, start=1):
        source, name, path = picks[i]
        item = f"rate_{n:03d}.png"
        with Image.open(path) as image:
            image.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(out / "images" / item)
        sheet.append({"item": item, "class": name, "rating": ""})
        key.append({"item": item, "class": name, "source": source, "file": str(path)})
    pd.DataFrame(sheet).to_csv(out / "rating_sheet.csv", index=False)
    args.key_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(key).to_csv(args.key_dir / "answer_key.csv", index=False)
    print(f"Wrote {len(sheet)} crops and rating_sheet.csv to {out}; answer_key.csv to {args.key_dir}")
    return 0


def score(args: argparse.Namespace) -> int:
    sheet = pd.read_csv(args.sheet, dtype=str).fillna("")
    key = pd.read_csv(args.key, dtype=str)
    rated = key.merge(sheet[["item", "rating"]], on="item", validate="one_to_one")
    rating = rated["rating"].str.strip().str.lower()
    unknown = sorted(set(rating) - {"real", "generated"})
    if unknown:
        raise ValueError(f"Ratings must be 'real' or 'generated'; found {unknown}")
    rated["called_real"] = rating == "real"
    table = rated.groupby(["source", "class"])["called_real"].agg(["sum", "count"])
    table["share_called_real"] = table["sum"] / table["count"]
    print(table.to_string())
    print(rated.groupby("source")["called_real"].mean().rename("share_called_real").to_string())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return make(args) if args.command == "make" else score(args)


if __name__ == "__main__":
    sys.exit(main())
