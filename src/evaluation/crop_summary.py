#!/usr/bin/env python3
"""Per-crop ROI experiment (T18): one summary table of S25-S32, every crop scored on its own.

Run after ``evaluate.py --experiments-root <crop root>`` has scored ``crop/<family>/``. The table
copies each scenario's test metrics and case-bootstrap intervals from its ``test_metrics.json``;
there is no photo level and no comparison with other runs (plan: tickets/T18_crop_roi.md §2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import CROP_PATH, CROP_SCENARIO_IDS, FAMILY_ORDER  # noqa: E402
from src.evaluation.summarize import CI_METRICS, TABLE_METRICS  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summary table of the per-crop ROI run (S25-S32).")
    parser.add_argument("--crop-root", type=Path, required=True,
                        help="Run folder laid out as crop/<family>/, already evaluated by evaluate.py.")
    return parser.parse_args(argv)


def summary_row(family: str, metrics: dict) -> dict:
    row = {"scenario_id": CROP_SCENARIO_IDS[(CROP_PATH, family)], "family": family,
           "test_crops": metrics["n_images"], "test_cases": metrics["n_cases"],
           **{name: metrics[name] for name in TABLE_METRICS}}
    for name in CI_METRICS:
        row[f"{name}_ci_low"] = metrics["confidence_intervals"][name]["low"]
        row[f"{name}_ci_high"] = metrics["confidence_intervals"][name]["high"]
    return row


def summary_table(root: Path) -> pd.DataFrame:
    rows = []
    for family in FAMILY_ORDER:
        path = root / CROP_PATH / family / "test_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; train every S25-S32 scenario and run evaluate.py first")
        rows.append(summary_row(family, json.loads(path.read_text(encoding="utf-8"))))
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    table = summary_table(args.crop_root)
    out = args.crop_root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "crop_summary_table.csv", index=False)
    for r in table.to_dict("records"):
        print(f"{r['scenario_id']} {r['family']:<20} crop macro-F1 {r['macro_f1']:.3f} "
              f"[{r['macro_f1_ci_low']:.3f}, {r['macro_f1_ci_high']:.3f}] ({r['test_crops']} crops, "
              f"{r['test_cases']} cases)", flush=True)
    print(f"Wrote {out / 'crop_summary_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
