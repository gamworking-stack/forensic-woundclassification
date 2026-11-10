#!/usr/bin/env python3
"""T34: choose the generated crops that join a classifier's training split (``--extra-train``).

Each chosen class is topped up to ``--target`` training crops (default: the median class count of the
real training crops), with generated crops the memorisation check did not flag, drawn at random with
``--seed``. A class already at or above the target gets none. Writes a CSV (absolute ``path``,
``class``, ``generator``, ``seed``) for ``train.py``/``cnn_svm.py --extra-train``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.generate_crops import SMALL_CLASSES  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generated crops to add to a training split.")
    parser.add_argument("--samples", type=Path, required=True, help="Folder written by generate_crops.py.")
    parser.add_argument("--nearest", type=Path, required=True, help="nearest.csv of generated_quality.py.")
    parser.add_argument("--train-dir", type=Path, required=True,
                        help="Generator dataset (crops.csv): the real training crops the generator saw.")
    parser.add_argument("--classes", nargs="+", default=list(SMALL_CLASSES))
    parser.add_argument("--target", type=int, default=None, help="Training crops per class after topping up.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    counts = pd.read_csv(args.train_dir / "crops.csv")["class"].value_counts()
    target = args.target if args.target is not None else int(np.median(counts.to_numpy()))
    samples = pd.read_csv(args.samples / "samples.csv")
    flagged = set(pd.read_csv(args.nearest).query("flagged")["file"])
    rng = np.random.default_rng(args.seed)

    rows = []
    for name in args.classes:
        need = max(target - int(counts.get(name, 0)), 0)
        pool = samples[samples["class"] == name]
        pool = pool[[str(args.samples / f) not in flagged for f in pool["file"]]]
        if need > len(pool):
            raise ValueError(f"{name}: {need} generated crops needed, only {len(pool)} unflagged in {args.samples}")
        chosen = pool.iloc[rng.choice(len(pool), need, replace=False)] if need else pool.iloc[:0]
        rows += [{"path": str((args.samples / f).resolve()), "class": name, "generator": g, "seed": s}
                 for f, g, s in zip(chosen["file"], chosen["generator"], chosen["seed"])]
        print(f"{name}: {int(counts.get(name, 0))} real + {need} generated = {int(counts.get(name, 0)) + need}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["path", "class", "generator", "seed"]).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} generated crops (target {target} per class) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
