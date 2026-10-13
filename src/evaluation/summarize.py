#!/usr/bin/env python3
"""One table of all 16 scenarios and the paired full vs ROI comparison on the same test cases (T15).

Every bootstrap resample draws one set of test cases and scores both input paths of all eight
families on it, so each ROI - full difference is paired by case. The plan (metrics, resamples,
p-value, Holm adjustment) is fixed in tickets/T15_summary_and_full_vs_roi.md §2.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import FAMILY_ORDER, INPUT_PATHS, SCENARIO_IDS  # noqa: E402
from src.evaluation.metrics import Predictions, load_predictions, summary_metrics  # noqa: E402
from src.evaluation.plots import plot_full_vs_roi  # noqa: E402


METRICS = ("macro_f1", "balanced_accuracy", "macro_auc")  # primary first
TABLE_METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc")
CI_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")
POOLED = "all families (mean)"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summary table and paired full vs ROI comparison.")
    parser.add_argument("--experiments-root", type=Path, required=True,
                        help="A run folder laid out as <full|roi>/<family>/, already evaluated by T08.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Paired case-level resamples.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def scenario_folder(root: Path, input_path: str, family: str) -> Path:
    folder = root / input_path / family
    for name in ("test_predictions.csv", "test_metrics.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"{folder} has no {name}; run evaluate.py on every scenario first")
    return folder


def summary_table(root: Path) -> pd.DataFrame:
    """One row per scenario in S01-S16 order, copied from T08's test_metrics.json."""
    rows = []
    for (input_path, family), scenario_id in sorted(SCENARIO_IDS.items(), key=lambda kv: kv[1]):
        folder = scenario_folder(root, input_path, family)
        m = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
        row = {"scenario_id": scenario_id, "family": family, "input_path": input_path,
               "test_images": m["n_images"], "test_cases": m["n_cases"],
               **{name: m[name] for name in TABLE_METRICS}}
        for name in CI_METRICS:
            row[f"{name}_ci_low"] = m["confidence_intervals"][name]["low"]
            row[f"{name}_ci_high"] = m["confidence_intervals"][name]["high"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_pairs(root: Path) -> Dict[str, Tuple[Predictions, Predictions]]:
    """(full, roi) test predictions per family; all 16 must share the same test cases."""
    pairs = {family: tuple(load_predictions(scenario_folder(root, path, family) / "test_predictions.csv", "test")
                           for path in INPUT_PATHS)
             for family in FAMILY_ORDER}
    reference = set(pairs[FAMILY_ORDER[0]][0].cases)
    for family, preds in pairs.items():
        for path, p in zip(INPUT_PATHS, preds):
            if set(p.cases) != reference:
                raise ValueError(f"{path}/{family} was tested on other cases than "
                                 f"{INPUT_PATHS[0]}/{FAMILY_ORDER[0]}; paired comparison is undefined")
    return pairs


def bootstrap_p(draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a zero difference."""
    return float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))


def holm(p_values: List[float]) -> List[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def paired_comparison(pairs: Dict[str, Tuple[Predictions, Predictions]],
                      n_resamples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """ROI - full per family and metric, with paired case-level bootstrap intervals and p-values."""
    families = list(pairs)
    case_ids = np.array(sorted(set(pairs[families[0]][0].cases)))
    rows_of = {(f, k): [np.flatnonzero(pairs[f][k].cases == c) for c in case_ids]
               for f in families for k in (0, 1)}

    def scores(f: str, k: int, idx: Optional[np.ndarray] = None) -> Dict[str, float]:
        p = pairs[f][k]
        if idx is None:
            return summary_metrics(p.y_true, p.y_pred, p.probs)
        return summary_metrics(p.y_true[idx], p.y_pred[idx], p.probs[idx])

    observed = {f: (scores(f, 0), scores(f, 1)) for f in families}
    draws = {(f, m): [] for f in families for m in METRICS}
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_resamples):
            picked = rng.integers(0, len(case_ids), size=len(case_ids))
            for f in families:
                full, roi = (scores(f, k, np.concatenate([rows_of[(f, k)][i] for i in picked])) for k in (0, 1))
                for m in METRICS:
                    draws[(f, m)].append(roi[m] - full[m])

    rows = []
    for m in METRICS:
        family_rows = []
        for f in families:
            d = np.array(draws[(f, m)])
            d = d[~np.isnan(d)]
            full, roi = observed[f][0][m], observed[f][1][m]
            family_rows.append({"family": f, "metric": m, "full": full, "roi": roi, "difference": roi - full,
                                "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                                "p_value": bootstrap_p(d), "n_cases": len(case_ids), "resamples": len(d)})
        for row, adjusted in zip(family_rows, holm([r["p_value"] for r in family_rows])):
            row["p_holm"] = adjusted
        pooled = np.nanmean(np.array([draws[(f, m)] for f in families]), axis=0)
        pooled = pooled[~np.isnan(pooled)]
        rows += family_rows + [{
            "family": POOLED, "metric": m,
            "full": float(np.mean([r["full"] for r in family_rows])),
            "roi": float(np.mean([r["roi"] for r in family_rows])),
            "difference": float(np.mean([r["difference"] for r in family_rows])),
            "ci_low": float(np.percentile(pooled, 2.5)), "ci_high": float(np.percentile(pooled, 97.5)),
            "p_value": bootstrap_p(pooled), "p_holm": np.nan, "n_cases": len(case_ids), "resamples": len(pooled),
        }]
    columns = ["family", "metric", "full", "roi", "difference", "ci_low", "ci_high",
               "p_value", "p_holm", "n_cases", "resamples"]
    return pd.DataFrame(rows, columns=columns)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = args.experiments_root
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    table = summary_table(root)
    table.to_csv(out / "summary_table.csv", index=False)
    comparison = paired_comparison(load_pairs(root), args.bootstrap, args.seed)
    comparison.to_csv(out / "full_vs_roi.csv", index=False)
    plot_full_vs_roi(comparison[comparison["metric"] == "macro_f1"], POOLED,
                     "Test macro-F1: ROI minus full image", out / "full_vs_roi_macro_f1.png")

    for row in comparison[comparison["metric"] == "macro_f1"].to_dict("records"):
        print(f"{row['family']:<22} ROI-full {row['difference']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] "
              f"p={row['p_value']:.3f} holm={row['p_holm']:.3f}", flush=True)
    print(f"Wrote summary_table.csv, full_vs_roi.csv and full_vs_roi_macro_f1.png to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
