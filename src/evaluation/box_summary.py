#!/usr/bin/env python3
"""Per-box ROI experiment (T17): photo-level scores and paired comparison with the main run.

Run after ``evaluate.py --experiments-root <box root>`` has scored the crop-level files of
S17-S24. For every scenario this script averages each photo's crop probabilities into
``box/<family>/photo/<split>_predictions.csv`` and scores them with the T08 code. It then compares
per-box (photo level) with the main run's union ROI and full image on the same test cases, with
the T15 paired case-level bootstrap. The plan is fixed in tickets/T17_box_roi.md §2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import BOX_PATH, BOX_SCENARIO_IDS, FAMILY_ORDER, SPLITS, evaluate_scenario  # noqa: E402
from src.evaluation.metrics import Predictions, load_predictions  # noqa: E402
from src.evaluation.plots import plot_full_vs_roi  # noqa: E402
from src.evaluation.summarize import CI_METRICS, POOLED, TABLE_METRICS, load_pairs, paired_comparison  # noqa: E402
from src.training.photo import photo_prediction_frame  # noqa: E402


CROP_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photo-level scores of the per-box ROI run and paired comparison.")
    parser.add_argument("--box-root", type=Path, required=True,
                        help="Run folder laid out as box/<family>/, already evaluated by evaluate.py.")
    parser.add_argument("--main-root", type=Path, required=True,
                        help="Main run (full/ and roi/ for all 8 families), already evaluated.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Paired case-level resamples.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def box_folder(root: Path, family: str) -> Path:
    folder = root / BOX_PATH / family
    for name in ("validation_predictions.csv", "test_predictions.csv", "test_metrics.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"{folder} has no {name}; train every S17-S24 scenario and run evaluate.py first")
    return folder


def write_photo_level(folder: Path, title: str, n_resamples: int, seed: int) -> Dict:
    """Photo-level prediction files and metric files in ``folder/photo``; return the test metrics."""
    photo_dir = folder / "photo"
    photo_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        crops = pd.read_csv(folder / f"{split}_predictions.csv", dtype=str)
        frame = photo_prediction_frame(crops.astype({c: float for c in crops.columns if c.startswith("prob_")}))
        frame.to_csv(photo_dir / f"{split}_predictions.csv", index=False)
    return evaluate_scenario(photo_dir, title, n_resamples, seed)


def summary_row(family: str, photo: Dict, crop: Dict) -> Dict:
    row = {"scenario_id": BOX_SCENARIO_IDS[(BOX_PATH, family)], "family": family,
           "test_photos": photo["n_images"], "test_cases": photo["n_cases"],
           **{name: photo[name] for name in TABLE_METRICS}}
    for name in CI_METRICS:
        row[f"{name}_ci_low"] = photo["confidence_intervals"][name]["low"]
        row[f"{name}_ci_high"] = photo["confidence_intervals"][name]["high"]
    row["test_crops"] = crop["n_images"]
    row.update({f"crop_{name}": crop[name] for name in CROP_METRICS})
    return row


def compare(reference: Dict[str, Predictions], box: Dict[str, Predictions], n_resamples: int, seed: int) -> pd.DataFrame:
    """Box - reference per family with the T15 paired bootstrap; columns renamed to reference/box."""
    table = paired_comparison({f: (reference[f], box[f]) for f in FAMILY_ORDER}, n_resamples, seed)
    return table.rename(columns={"full": "reference", "roi": "box"})


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.box_root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    rows, box = [], {}
    for family in FAMILY_ORDER:
        folder = box_folder(args.box_root, family)
        title = f"{BOX_SCENARIO_IDS[(BOX_PATH, family)]} {family} (box, photo level)"
        photo = write_photo_level(folder, title, args.bootstrap, args.seed)
        crop = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
        rows.append(summary_row(family, photo, crop))
        box[family] = load_predictions(folder / "photo" / "test_predictions.csv", "test")
    pd.DataFrame(rows).to_csv(out / "box_summary_table.csv", index=False)

    main_pairs = load_pairs(args.main_root)  # checks that all 16 main scenarios share their test cases
    reference = main_pairs[FAMILY_ORDER[0]][1].signature()
    for family, p in box.items():
        if p.signature() != reference:
            raise ValueError(f"box/{family} photo-level test set differs from the main run's "
                             "(cases or images per class); paired comparison is undefined")

    versus_union = compare({f: main_pairs[f][1] for f in FAMILY_ORDER}, box, args.bootstrap, args.seed)
    versus_full = compare({f: main_pairs[f][0] for f in FAMILY_ORDER}, box, args.bootstrap, args.seed)
    versus_union.to_csv(out / "box_vs_union.csv", index=False)
    versus_full.to_csv(out / "box_vs_full.csv", index=False)
    plot_full_vs_roi(versus_union[versus_union["metric"] == "macro_f1"], POOLED,
                     "Test macro-F1 (photo level): per-box minus union ROI", out / "box_vs_union_macro_f1.png",
                     labels=("union ROI", "per-box ROI"))

    for name, table in (("box-union", versus_union), ("box-full", versus_full)):
        for row in table[table["metric"] == "macro_f1"].to_dict("records"):
            holm = "" if pd.isna(row["p_holm"]) else f" holm={row['p_holm']:.3f}"
            print(f"{row['family']:<22} {name} {row['difference']:+.3f} [{row['ci_low']:+.3f}, "
                  f"{row['ci_high']:+.3f}] p={row['p_value']:.3f}{holm}", flush=True)
    print(f"Wrote box_summary_table.csv, box_vs_union.csv, box_vs_full.csv and box_vs_union_macro_f1.png to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
