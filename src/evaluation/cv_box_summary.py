#!/usr/bin/env python3
"""Per-box ROI cross-validation (T17): photo-level fold scores and paired comparison with T12b.

The 5 held-out-fold prediction files of each S17-S24 scenario hold one row per crop. Each fold is
turned into photo-level predictions (mean of a photo's crop probabilities) and scored; the folds
are joined into ``box/<family>/oof_photo_predictions.csv``. The out-of-fold photos are compared
with the T12b out-of-fold predictions of the union ROI and full-image paths (``oof_predictions.csv``
written by cv_summary.py) on the same development cases, with the T15 paired bootstrap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.box_summary import compare  # noqa: E402
from src.evaluation.cv_summary import FOLD_METRICS, fold_predictions  # noqa: E402
from src.evaluation.evaluate import BOX_PATH, BOX_SCENARIO_IDS, FAMILY_ORDER, INPUT_PATHS  # noqa: E402
from src.evaluation.metrics import Predictions, compute_metrics, load_predictions, summary_metrics  # noqa: E402
from src.evaluation.plots import plot_full_vs_roi  # noqa: E402
from src.evaluation.summarize import POOLED  # noqa: E402
from src.training.photo import photo_prediction_frame  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the per-box 5-fold CV (T17) against T12b.")
    parser.add_argument("--cv-box-root", type=Path, required=True, help="Per-box CV run laid out as box/<family>/fold_<k>/.")
    parser.add_argument("--cv-main-root", type=Path, required=True,
                        help="T12b CV run, already summarized by cv_summary.py (oof_predictions.csv per scenario).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def photo_frame_of(path: Path) -> pd.DataFrame:
    crops = pd.read_csv(path, dtype=str)
    return photo_prediction_frame(crops.astype({c: float for c in crops.columns if c.startswith("prob_")}))


def box_out_of_fold(cv_root: Path, family: str, folds: int) -> Tuple[List[Dict[str, float]], Predictions]:
    """Per-fold photo-level summary metrics and the joined photo-level out-of-fold predictions."""
    frames, per_fold = [], []
    for path in fold_predictions(cv_root, BOX_PATH, family, folds):
        frame = photo_frame_of(path)
        frames.append(frame)
        tmp = path.parent / "validation_photo_predictions.csv"
        frame.to_csv(tmp, index=False)
        p = load_predictions(tmp, "validation")
        per_fold.append(summary_metrics(p.y_true, p.y_pred, p.probs))
    oof_path = cv_root / BOX_PATH / family / "oof_photo_predictions.csv"
    pd.concat(frames, ignore_index=True).to_csv(oof_path, index=False)
    return per_fold, load_predictions(oof_path, "validation")  # a photo in two folds raises here


def table_row(family: str, folds: int, per_fold: List[Dict[str, float]], oof: Predictions) -> Dict:
    pooled = compute_metrics(oof)
    row = {"scenario_id": BOX_SCENARIO_IDS[(BOX_PATH, family)], "family": family, "input_path": BOX_PATH,
           "folds": folds, "oof_photos": pooled["n_images"], "oof_cases": pooled["n_cases"]}
    for m in FOLD_METRICS:
        values = np.array([f[m] for f in per_fold])
        row[f"{m}_mean"], row[f"{m}_sd"] = float(values.mean()), float(values.std(ddof=1))
        row[f"{m}_oof"] = pooled[m]
        row[f"{m}_folds"] = " ".join(f"{v:.4f}" for v in values)
    return row


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.cv_box_root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    rows, box, reference = [], {}, {path: {} for path in INPUT_PATHS}
    for family in FAMILY_ORDER:
        per_fold, oof = box_out_of_fold(args.cv_box_root, family, args.folds)
        rows.append(table_row(family, args.folds, per_fold, oof))
        box[family] = oof
        for path in INPUT_PATHS:
            main_oof = args.cv_main_root / path / family / "oof_predictions.csv"
            if not main_oof.is_file():
                raise FileNotFoundError(f"{main_oof} not found; run cv_summary.py on the T12b run first")
            reference[path][family] = load_predictions(main_oof, "validation")

    signature = reference["roi"][FAMILY_ORDER[0]].signature()
    for family, p in box.items():
        if p.signature() != signature:
            raise ValueError(f"box/{family} out-of-fold photos differ from the T12b union ROI run "
                             "(cases or images per class); paired comparison is undefined")

    pd.DataFrame(rows).to_csv(out / "cv_box_table.csv", index=False)
    versus_union = compare(reference["roi"], box, args.bootstrap, args.seed)
    versus_full = compare(reference["full"], box, args.bootstrap, args.seed)
    versus_union.to_csv(out / "cv_box_vs_union.csv", index=False)
    versus_full.to_csv(out / "cv_box_vs_full.csv", index=False)
    plot_full_vs_roi(versus_union[versus_union["metric"] == "macro_f1"], POOLED,
                     "CV out-of-fold macro-F1 (photo level): per-box minus union ROI",
                     out / "cv_box_vs_union_macro_f1.png", labels=("union ROI", "per-box ROI"))

    for r in rows:
        print(f"{r['scenario_id']} {r['family']:<20} box macro-F1 {r['macro_f1_mean']:.3f} ± {r['macro_f1_sd']:.3f} "
              f"(out of fold {r['macro_f1_oof']:.3f}, {r['oof_cases']} cases)", flush=True)
    for name, table in (("box-union", versus_union), ("box-full", versus_full)):
        for r in table[table["metric"] == "macro_f1"].to_dict("records"):
            holm = "" if pd.isna(r["p_holm"]) else f" holm={r['p_holm']:.3f}"
            print(f"{r['family']:<22} {name} {r['difference']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
                  f"p={r['p_value']:.3f}{holm}", flush=True)
    print(f"Wrote cv_box_table.csv, cv_box_vs_union.csv, cv_box_vs_full.csv and the figure to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
