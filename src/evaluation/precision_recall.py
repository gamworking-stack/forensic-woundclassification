#!/usr/bin/env python3
"""T28: macro precision and macro recall with case-bootstrap intervals, for the manuscript's Table 2.

T08's bootstrap gives intervals for accuracy, balanced accuracy, macro-F1 and macro AUC only. This
draws the very same resamples (``bootstrap_draws``: sorted cases, ``default_rng(seed)``, one
``integers`` call per resample) and scores macro precision and macro recall on each, over the classes
present in the resample, unpredicted classes counting 0, as ``compute_metrics`` does. Macro recall is
balanced accuracy, so its interval must equal the saved balanced-accuracy interval; the script checks
that. The shared bootstrap is not changed, so saved runs and their verification stay as they are.

Run it on prediction copies whose ``pseudonimized_id`` is the patient (T25) for patient-level intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import CROP_PATH, CROP_SCENARIO_IDS, FAMILY_ORDER, SCENARIO_IDS  # noqa: E402
from src.evaluation.metrics import Predictions, load_predictions  # noqa: E402

METRICS = ("macro_precision", "macro_recall")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Macro precision and recall with case-bootstrap intervals.")
    parser.add_argument("--main-root", type=Path, required=True, help="Evaluated main run (full/<family>/).")
    parser.add_argument("--crop-root", type=Path, required=True, help="Evaluated per-crop run (crop/<family>/).")
    parser.add_argument("--output", type=Path, required=True, help="CSV to write (outside the repository).")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Macro precision and recall over the classes present in ``y_true``."""
    present = np.unique(y_true)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    return np.array([precision.mean(), recall.mean()])


def intervals(p: Predictions, n_resamples: int, seed: int) -> Dict[str, float]:
    """Observed values and 95% percentile intervals, on ``bootstrap_draws``' resamples."""
    rng = np.random.default_rng(seed)
    case_ids = np.array(sorted(set(p.cases)))
    rows_of = [np.flatnonzero(p.cases == c) for c in case_ids]
    draws = np.empty((n_resamples, len(METRICS)))
    for b in range(n_resamples):
        picked = rng.integers(0, len(case_ids), size=len(case_ids))
        idx = np.concatenate([rows_of[i] for i in picked])
        draws[b] = precision_recall(p.y_true[idx], p.y_pred[idx])
    observed = precision_recall(p.y_true, p.y_pred)
    row = {}
    for j, name in enumerate(METRICS):
        row.update({name: float(observed[j]), f"{name}_ci_low": float(np.percentile(draws[:, j], 2.5)),
                    f"{name}_ci_high": float(np.percentile(draws[:, j], 97.5))})
    return row


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rows = []
    for input_path, root, ids in (("full", args.main_root, SCENARIO_IDS), (CROP_PATH, args.crop_root, CROP_SCENARIO_IDS)):
        for family in FAMILY_ORDER:
            folder = root / input_path / family
            p = load_predictions(folder / "test_predictions.csv", "test")
            row = {"scenario_id": ids[(input_path, family)], "input_path": input_path, "family": family,
                   "test_rows": len(p.y_true), "test_cases": len(set(p.cases)), **intervals(p, args.bootstrap, args.seed)}
            # Macro recall is balanced accuracy: same value and, on the same resamples, the same interval
            saved = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
            ci = saved["confidence_intervals"]["balanced_accuracy"]
            if not np.allclose([row["macro_recall"], row["macro_recall_ci_low"], row["macro_recall_ci_high"]],
                               [saved["balanced_accuracy"], ci["low"], ci["high"]]):
                raise ValueError(f"{folder}: macro recall does not reproduce the saved balanced accuracy and its interval")
            if not np.isclose(row["macro_precision"], saved["macro_precision"]):
                raise ValueError(f"{folder}: macro precision differs from test_metrics.json")
            rows.append(row)
            print(f"{row['scenario_id']} {family:<20} precision {row['macro_precision']:.3f} "
                  f"[{row['macro_precision_ci_low']:.3f}, {row['macro_precision_ci_high']:.3f}]  recall "
                  f"{row['macro_recall']:.3f} [{row['macro_recall_ci_low']:.3f}, {row['macro_recall_ci_high']:.3f}]",
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
