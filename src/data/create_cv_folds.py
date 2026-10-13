#!/usr/bin/env python3
"""Assign the development cases (train + validation) to 5 case-grouped folds, once (T12b).

Every case lands in exactly one fold and test cases are excluded. A seeded search over random
assignments keeps the one closest to equal class shares per fold, with every class present in
every fold, as in the T02 split. Balance is scored on the ROI manifest's development rows, the
matched images both input paths train on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_case_split import build_group_matrices  # noqa: E402


GROUP_COLUMN = "pseudonimized_id"
LABEL_COLUMN = "canonical_wound_label"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case-grouped CV folds over the development data.")
    parser.add_argument("--split-manifest", type=Path, default=Path("data/splits/case_level_split_manifest.csv"))
    parser.add_argument("--roi-manifest", type=Path, default=Path("data/roi/roi_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=50_000)
    return parser.parse_args(argv)


def development_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[df["split"].str.strip().str.lower() != "test"].reset_index(drop=True)


def fold_score(assignment: np.ndarray, images: np.ndarray, presence: np.ndarray, k: int) -> float:
    """Quadratic deviation from equal class shares per fold; inf if a fold misses a class."""
    observed_images = np.vstack([images[assignment == f].sum(axis=0) for f in range(k)])
    observed_presence = np.vstack([presence[assignment == f].sum(axis=0) for f in range(k)])
    if (observed_images == 0).any() or (observed_presence == 0).any():
        return float("inf")
    expected_images = images.sum(axis=0) / k
    expected_presence = presence.sum(axis=0) / k
    return float((np.square(observed_images - expected_images) / (expected_images + 1.0)).sum()
                 + (np.square(observed_presence - expected_presence) / (expected_presence + 1.0)).sum())


def assign_folds(cases: List[str], scored: pd.DataFrame, k: int, seed: int, trials: int) -> Dict[str, int]:
    """Best of ``trials`` random equal-size assignments of ``cases`` to folds 1..k."""
    group_ids, _, images, presence = build_group_matrices(scored, GROUP_COLUMN, LABEL_COLUMN)
    index = {case: i for i, case in enumerate(cases)}
    rows = np.array([index[c] for c in group_ids])  # scored cases, in matrix order
    rng = np.random.default_rng(seed)
    base = np.arange(len(cases)) % k
    best, best_score = None, float("inf")
    for _ in range(trials):
        assignment = rng.permutation(base)
        score = fold_score(assignment[rows], images, presence, k)
        if score < best_score:
            best, best_score = assignment.copy(), score
    if best is None:
        raise RuntimeError("No assignment placed every class in every fold; increase --trials")
    return {case: int(best[i]) + 1 for case, i in index.items()}


def fold_report(scored: pd.DataFrame, folds: Dict[str, int]) -> Dict[str, Any]:
    df = scored.assign(fold=scored[GROUP_COLUMN].map(folds))
    return {
        str(f): {
            "cases": int(part[GROUP_COLUMN].nunique()),
            "images": int(len(part)),
            "images_per_class": {c: int(n) for c, n in part[LABEL_COLUMN].value_counts().sort_index().items()},
            "cases_per_class": {c: int(n) for c, n in part.groupby(LABEL_COLUMN)[GROUP_COLUMN].nunique().items()},
        }
        for f, part in df.groupby("fold")
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    full = development_rows(args.split_manifest)
    roi = development_rows(args.roi_manifest)
    cases = sorted(set(full[GROUP_COLUMN]) | set(roi[GROUP_COLUMN]), key=lambda c: (len(c), c))

    folds = assign_folds(cases, roi, args.folds, args.seed, args.trials)
    test_cases = set(pd.read_csv(args.split_manifest, dtype=str, keep_default_na=False)
                     .query("split == 'test'")[GROUP_COLUMN])
    assert not test_cases & set(folds), "a test case was assigned to a fold"
    report = {"folds": args.folds, "seed": args.seed, "trials": args.trials,
              "development_cases": len(folds), "scored_on": str(args.roi_manifest),
              "roi_rows": fold_report(roi, folds), "full_rows": fold_report(full, folds), "frozen": True}
    for path_rows in ("roi_rows", "full_rows"):
        for f, part in report[path_rows].items():
            if len(part["images_per_class"]) != 7:
                raise AssertionError(f"fold {f} misses a class in {path_rows}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({GROUP_COLUMN: list(folds), "fold": list(folds.values())}) \
        .to_csv(args.output_dir / "cv_folds.csv", index=False)
    with (args.output_dir / "cv_folds_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for f, part in report["roi_rows"].items():
        print(f"fold {f}: {part['cases']} cases, {part['images']} ROI images, classes {part['cases_per_class']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
