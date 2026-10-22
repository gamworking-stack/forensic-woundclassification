#!/usr/bin/env python3
"""Per-crop run (T22): one crop per photo, the most confident one, scored like the full image.

Run after ``evaluate.py --experiments-root <crop root>`` has scored ``crop/<family>/`` (S25-S32).
For every family this script keeps, per photo, the crop with the highest top-class probability
(no label is used) and writes ``crop/<family>/top_crop/<split>_predictions.csv``, scored with the
T08 code. Each of the 157 test photos then has one full-image and one crop prediction. The summary
table lists both inputs side by side with their own case-bootstrap intervals; there is no paired
test (plan: tickets/T22_top_crop.md §2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import CROP_PATH, CROP_SCENARIO_IDS, FAMILY_ORDER, SCENARIO_IDS, SPLITS, evaluate_scenario  # noqa: E402
from src.evaluation.metrics import load_predictions  # noqa: E402
from src.evaluation.summarize import CI_METRICS, TABLE_METRICS, scenario_folder  # noqa: E402
from src.training.photo import PHOTO_COLUMN, photo_top_crop_frame  # noqa: E402


TOP = "top_crop"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One crop per photo (the most confident) for the per-crop run.")
    parser.add_argument("--crop-root", type=Path, required=True,
                        help="Per-crop run laid out as crop/<family>/, already evaluated.")
    parser.add_argument("--main-root", type=Path, required=True,
                        help="Main run (full/<family>/), already evaluated, for the full-image rows.")
    parser.add_argument("--box-manifest", type=Path, default=Path("data/roi_box/roi_manifest.csv"))
    parser.add_argument("--bootstrap", type=int, default=2000, help="Case-level resamples for test CIs.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def crop_photos(box_manifest: Path) -> Dict[str, str]:
    """Crop file -> its photo, from the per-box manifest (crop prediction files name only the crop)."""
    rows = pd.read_csv(box_manifest, dtype=str, keep_default_na=False)
    return dict(zip(rows["roi_relative_path"], rows["relative_path"]))


def write_top_crop(folder: Path, photo_of: Dict[str, str]) -> Path:
    """``folder/top_crop/<split>_predictions.csv`` with each photo's most confident crop."""
    out = folder / TOP
    out.mkdir(exist_ok=True)
    for split in SPLITS:
        # Name every crop's photo; a crop missing from the manifest is an error
        crops = pd.read_csv(folder / f"{split}_predictions.csv", dtype=str)
        crops[PHOTO_COLUMN] = crops["relative_path"].map(photo_of)
        if crops[PHOTO_COLUMN].isna().any():
            missing = crops.loc[crops[PHOTO_COLUMN].isna(), "relative_path"].tolist()
            raise ValueError(f"{folder}/{split}: {len(missing)} crops not in the per-box manifest, e.g. {missing[:3]}")
        photo_top_crop_frame(crops).to_csv(out / f"{split}_predictions.csv", index=False)
    return out


def table_row(scenario_id: str, family: str, input_name: str, metrics: Dict) -> Dict:
    # Photo count, headline metrics and the low/high bound of each interval
    row = {"scenario_id": scenario_id, "family": family, "input": input_name,
           "test_photos": metrics["n_images"], "test_cases": metrics["n_cases"],
           **{name: metrics[name] for name in TABLE_METRICS}}
    for name in CI_METRICS:
        row[f"{name}_ci_low"] = metrics["confidence_intervals"][name]["low"]
        row[f"{name}_ci_high"] = metrics["confidence_intervals"][name]["high"]
    return row


def class_rows(report: Path, family: str, input_name: str) -> pd.DataFrame:
    # The per-class lines of a T08 classification report, without the averages
    frame = pd.read_csv(report)
    frame = frame[~frame["class"].isin(["macro avg", "weighted avg"])]
    return frame.assign(family=family, input=input_name)[["family", "input", "class", "precision", "recall", "f1",
                                                          "auc", "support_images", "support_cases"]]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.crop_root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    photo_of = crop_photos(args.box_manifest)

    rows, classes = [], []
    for family in FAMILY_ORDER:
        # Full image: the main run's scored test files
        full = scenario_folder(args.main_root, "full", family)
        rows.append(table_row(SCENARIO_IDS[("full", family)], family, "full image",
                              json.loads((full / "test_metrics.json").read_text(encoding="utf-8"))))
        classes.append(class_rows(full / "test_classification_report.csv", family, "full image"))
        # Top crop: one crop per photo from the per-crop run, scored with the T08 code
        folder = args.crop_root / CROP_PATH / family
        top = write_top_crop(folder, photo_of)
        title = f"{CROP_SCENARIO_IDS[(CROP_PATH, family)]} {family} (top crop per photo)"
        rows.append(table_row(CROP_SCENARIO_IDS[(CROP_PATH, family)], family, "top crop",
                              evaluate_scenario(top, title, args.bootstrap, args.seed)))
        classes.append(class_rows(top / "test_classification_report.csv", family, "top crop"))
        # Both inputs must cover the same test photos per case and class
        if load_predictions(top / "test_predictions.csv", "test").signature() != \
                load_predictions(full / "test_predictions.csv", "test").signature():
            raise ValueError(f"{family}: top-crop test photos differ from the full-image test photos")

    table = pd.DataFrame(rows)
    table.to_csv(out / "top_crop_table.csv", index=False)
    pd.concat(classes).to_csv(out / "top_crop_per_class.csv", index=False)
    for r in table.to_dict("records"):
        print(f"{r['scenario_id']} {r['family']:<20} {r['input']:<10} macro-F1 {r['macro_f1']:.3f} "
              f"[{r['macro_f1_ci_low']:.3f}, {r['macro_f1_ci_high']:.3f}] accuracy {r['accuracy']:.3f} "
              f"({r['test_photos']} photos, {r['test_cases']} cases)", flush=True)
    print(f"Wrote top_crop_table.csv and top_crop_per_class.csv to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
