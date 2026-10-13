#!/usr/bin/env python3
"""Summarize the T12b cross-validation: mean ± SD over folds and paired full vs ROI out of fold.

Each scenario's 5 held-out-fold prediction files are scored per fold, then joined into one
out-of-fold file covering every development case once. The paired comparison is the T15 method
(summarize.paired_comparison) applied to those out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import FAMILY_ORDER, INPUT_PATHS, SCENARIO_IDS  # noqa: E402
from src.evaluation.metrics import Predictions, compute_metrics, load_predictions, summary_metrics  # noqa: E402
from src.evaluation.plots import plot_full_vs_roi  # noqa: E402
from src.evaluation.summarize import POOLED, paired_comparison  # noqa: E402


FOLD_METRICS = ("macro_f1", "balanced_accuracy", "macro_auc", "accuracy")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the 5-fold CV runs (T12b).")
    parser.add_argument("--cv-root", type=Path, required=True, help="CV run folder laid out as <full|roi>/<family>/fold_<k>/.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def fold_predictions(cv_root: Path, input_path: str, family: str, folds: int) -> List[Path]:
    paths = [cv_root / input_path / family / f"fold_{k}" / "validation_predictions.csv" for k in range(1, folds + 1)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{input_path}/{family} is missing held-out-fold predictions: {missing}")
    return paths


def out_of_fold(cv_root: Path, input_path: str, family: str, folds: int) -> Tuple[List[Dict[str, float]], Predictions]:
    """Per-fold summary metrics and the joined out-of-fold predictions (written next to the folds)."""
    paths = fold_predictions(cv_root, input_path, family, folds)
    per_fold = []
    for p in paths:
        pred = load_predictions(p, "validation")
        per_fold.append(summary_metrics(pred.y_true, pred.y_pred, pred.probs))
    joined = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    oof_path = cv_root / input_path / family / "oof_predictions.csv"
    joined.to_csv(oof_path, index=False)
    return per_fold, load_predictions(oof_path, "validation")  # duplicate images across folds raise here


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.cv_root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    rows, pairs = [], {}
    for family in FAMILY_ORDER:
        oof = []
        for input_path in INPUT_PATHS:
            per_fold, pred = out_of_fold(args.cv_root, input_path, family, args.folds)
            pooled = compute_metrics(pred)
            row = {"scenario_id": SCENARIO_IDS[(input_path, family)], "family": family, "input_path": input_path,
                   "folds": args.folds, "oof_images": pooled["n_images"], "oof_cases": pooled["n_cases"]}
            for m in FOLD_METRICS:
                values = np.array([f[m] for f in per_fold])
                row[f"{m}_mean"], row[f"{m}_sd"] = float(values.mean()), float(values.std(ddof=1))
                row[f"{m}_oof"] = pooled[m]
                row[f"{m}_folds"] = " ".join(f"{v:.4f}" for v in values)
            rows.append(row)
            oof.append(pred)
        pairs[family] = tuple(oof)

    reference = set(pairs[FAMILY_ORDER[0]][0].cases)
    for family, preds in pairs.items():
        for input_path, p in zip(INPUT_PATHS, preds):
            if set(p.cases) != reference:
                raise ValueError(f"{input_path}/{family} out-of-fold cases differ from "
                                 f"{INPUT_PATHS[0]}/{FAMILY_ORDER[0]}; paired comparison is undefined")

    table = pd.DataFrame(rows).sort_values("scenario_id")
    table.to_csv(out / "cv_table.csv", index=False)
    comparison = paired_comparison(pairs, args.bootstrap, args.seed)
    comparison.to_csv(out / "cv_full_vs_roi.csv", index=False)
    plot_full_vs_roi(comparison[comparison["metric"] == "macro_f1"], POOLED,
                     "CV out-of-fold macro-F1: ROI minus full image", out / "cv_full_vs_roi_macro_f1.png")

    for r in table.to_dict("records"):
        print(f"{r['scenario_id']} {r['family']:<20} {r['input_path']:<4} macro-F1 {r['macro_f1_mean']:.3f} "
              f"± {r['macro_f1_sd']:.3f} (out of fold {r['macro_f1_oof']:.3f}, {r['oof_cases']} cases)", flush=True)
    for r in comparison[comparison["metric"] == "macro_f1"].to_dict("records"):
        print(f"{r['family']:<22} ROI-full {r['difference']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
              f"p={r['p_value']:.3f} holm={r['p_holm']:.3f}", flush=True)
    print(f"Wrote cv_table.csv, cv_full_vs_roi.csv and cv_full_vs_roi_macro_f1.png to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
