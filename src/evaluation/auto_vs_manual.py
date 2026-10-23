#!/usr/bin/env python3
"""Auto-box vs manual-box ROI: the T19 classification analyses (§2.7).

Layout under ``--experiments-dir`` (Raptor ``/project/a260012_FORWCL/experiments``):
- ``main/{full,roi}/<family>`` and ``roi_box/box/<family>``: the reported manual-box runs.
- ``auto_<arm>/detection/selection.json``: the arm's frozen selection and gate.
- ``auto_<arm>/primary/{roi,box}/<family>``: existing models on the arm's auto crops (``predict_on_manifest.py``).
- ``auto_<arm>_union/roi/<family>``, ``auto_<arm>_box/box/<family>``: the retrained scenarios.

Analyses:
- **Primary**, per passing arm and design (union, per-box at photo level): non-inferiority of auto
  vs manual for the same models.
  - The pooled 8-family macro-F1 difference gets its paired case-bootstrap interval.
  - Not inferior if the lower bound is above ``-margin``; inferior if the upper bound is below it;
    inconclusive otherwise.
  - The one-sided p (share of resamples with a pooled difference at or below ``-margin``) is
    Holm-adjusted over all primary tests.
  - Per-family differences: two-sided p, Holm over 8 families.
- **Fallback sensitivity**: the same pooled difference without the photos whose auto boxes fell
  back. Descriptive only.
- **Secondary**, two-sided, Holm over 8 families: retrained auto − manual, and auto − full image.
- Per arm: fallback counts and the share of auto boxes overlapping no manual box.

Resamples, seed and p-values are the T15 ones (``summarize.paired_draws``).
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
from src.detection.boxes import iou_matrix  # noqa: E402
from src.evaluation.evaluate import FAMILY_ORDER  # noqa: E402
from src.evaluation.metrics import Predictions, load_predictions, summary_metrics  # noqa: E402
from src.evaluation.summarize import holm, paired_comparison, paired_draws  # noqa: E402
from src.training.photo import MANIFEST_PHOTO_COLUMN, photo_prediction_frame  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402


ARMS = ("4Z", "4P", "4F")
DESIGNS = {"union": "roi", "box": "box"}
MARGIN = 0.03
PRIMARY_METRIC = "macro_f1"


def photo_level(crop_file: Path, out_file: Path) -> Path:
    """Write the photo-level prediction file (mean of a photo's crops) of a per-box crop file."""
    # Average each photo's crop probabilities into one row per photo and save it
    crops = pd.read_csv(crop_file, dtype=str)
    frame = photo_prediction_frame(crops.astype({c: float for c in crops.columns if c.startswith("prob_")}))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_file, index=False)
    return out_file


def predictions(folder: Path, design: str, cache: Path) -> Predictions:
    """A scenario's test predictions at photo level (union files already are)."""
    path = folder / "test_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    # Per-box files hold one row per crop: turn them into photo level first
    if design == "box":
        path = photo_level(path, cache / "test_predictions.csv")
    return load_predictions(path, "test")


def family_set(folders: Dict[str, Path], design: str, cache: Path) -> Dict[str, Predictions]:
    return {f: predictions(folders[f], design, cache / f) for f in FAMILY_ORDER}


def pooled(draws: Dict[Tuple[str, str], List[float]], metric: str = PRIMARY_METRIC) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # On each resample, the mean difference over the 8 families
        values = np.nanmean(np.array([draws[(f, metric)] for f in FAMILY_ORDER]), axis=0)
    return values[~np.isnan(values)]


def verdict(low: float, high: float, margin: float) -> str:
    # Whole interval above -margin: not inferior; whole interval below: inferior; otherwise inconclusive
    if low > -margin:
        return "not inferior"
    if high < -margin:
        return "inferior"
    return "inconclusive"


def photo_keys(frame: pd.DataFrame, design: str, crop_to_photo: Dict[str, str]) -> np.ndarray:
    """The photo of every row: photo-level files name it; union files name the crop."""
    # Photo-level files name the photo; union files name the crop, so map crop -> photo
    if design == "box":
        return frame["relative_path"].to_numpy()
    return frame["relative_path"].map(crop_to_photo).to_numpy()


def subset_difference(pairs: Dict[str, Tuple[Predictions, Predictions]], design: str, keep: set,
                      manual_map: Dict[str, str], auto_map: Dict[str, str]) -> float:
    """Pooled macro-F1 difference (auto − manual) on the photos in ``keep``; no resampling."""
    diffs = []
    # For each family: macro-F1 of manual and of auto on the kept photos only, then the mean difference
    for manual, auto in pairs.values():
        scores = []
        for p, mapping in ((manual, manual_map), (auto, auto_map)):
            mask = np.isin(photo_keys(p.frame, design, mapping), list(keep))
            scores.append(summary_metrics(p.y_true[mask], p.y_pred[mask], p.probs[mask])[PRIMARY_METRIC])
        diffs.append(scores[1] - scores[0])
    return float(np.mean(diffs))


def no_overlap_share(box_manifest: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Share of the test split's detected auto boxes (no full-image fallback) that touch no manual box."""
    # For each detected test box: does it overlap any manual box of its photo at all?
    rows = box_manifest[(box_manifest["split"] == "test") & (box_manifest["fallback"] != "full_image")]
    truth = {p: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for p, g in reference.groupby(MANIFEST_PHOTO_COLUMN)}
    misses = [float(iou_matrix(r[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(float), truth[r[MANIFEST_PHOTO_COLUMN]]).max() == 0)
              for _, r in rows.iterrows()]
    return float(np.mean(misses)) if misses else float("nan")


def tag(table: pd.DataFrame, **columns) -> pd.DataFrame:
    # Put the arm and design columns first
    for name, value in reversed(list(columns.items())):
        table.insert(0, name, value)
    return table


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-box vs manual-box ROI analyses (T19 §2.7).")
    parser.add_argument("--experiments-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--manifest-root", type=Path, default=DATA_DIR / "roi_auto",
                        help="data/roi_auto: <arm>/{union,box}/roi_manifest.csv and reference_boxes.csv")
    parser.add_argument("--union-manifest", type=Path, default=DATA_DIR / "roi/roi_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Output folder and a cache for photo-level files
    exp, out = args.experiments_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "photo_level"
    # The reported runs: full image, manual union ROI and manual per-box ROI, all at photo level
    manual_folders = {"union": {f: exp / "main" / "roi" / f for f in FAMILY_ORDER},
                      "box": {f: exp / "roi_box" / "box" / f for f in FAMILY_ORDER}}
    full = family_set({f: exp / "main" / "full" / f for f in FAMILY_ORDER}, "union", cache / "full")
    manual = {d: family_set(manual_folders[d], d, cache / "manual" / d) for d in DESIGNS}
    # Crop -> photo map for the manual union crops, and the manual test boxes
    manual_union_map = pd.read_csv(args.union_manifest, dtype=str).set_index("roi_relative_path")[MANIFEST_PHOTO_COLUMN].to_dict()
    reference = pd.read_csv(args.manifest_root / "reference_boxes.csv", dtype={MANIFEST_PHOTO_COLUMN: str})
    reference = reference[reference["split"] == "test"]

    # One pass per detector arm
    primary, by_family, retrained, versus_full, arms = [], [], [], [], {}
    for arm in args.arms:
        # Record the arm's choice and gate; an arm that failed its gate is not compared
        selection = json.loads((exp / f"auto_{arm}" / "detection" / "selection.json").read_text(encoding="utf-8"))
        arms[arm] = {"chosen_config": selection["chosen_config"], "threshold": selection["threshold"],
                     "gate_recall": selection["threshold_detail"]["recall"], "gate_passed": selection["gate_passed"]}
        if not selection["gate_passed"]:
            continue
        # The arm's auto crops: which test photos fell back, and how many auto boxes miss every wound
        box_manifest = pd.read_csv(args.manifest_root / arm / "box" / "roi_manifest.csv", dtype={MANIFEST_PHOTO_COLUMN: str})
        union_manifest = pd.read_csv(args.manifest_root / arm / "union" / "roi_manifest.csv", dtype=str)
        test_photos = union_manifest[union_manifest["split"] == "test"]
        fallback_photos = set(test_photos.loc[test_photos["fallback"] != "none", MANIFEST_PHOTO_COLUMN])
        arms[arm].update({"test_fallback": test_photos["fallback"].value_counts().to_dict(),
                          "test_auto_boxes_touching_no_manual_box": no_overlap_share(box_manifest, reference)})
        auto_union_map = union_manifest.set_index("roi_relative_path")[MANIFEST_PHOTO_COLUMN].to_dict()

        # Union design, then per-box design
        for design, input_path in DESIGNS.items():
            # Primary: the same models on auto crops, paired with their manual-crop predictions
            auto = family_set({f: exp / f"auto_{arm}" / "primary" / input_path / f for f in FAMILY_ORDER},
                              design, cache / arm / "primary" / design)
            pairs = {f: (manual[design][f], auto[f]) for f in FAMILY_ORDER}
            # One set of case resamples, used for the per-family table and the pooled test
            drawn = paired_draws(pairs, args.bootstrap, args.seed)
            table = paired_comparison(pairs, args.bootstrap, args.seed, precomputed=drawn)
            by_family.append(tag(table.rename(columns={"full": "manual", "roi": "auto"}), arm=arm, design=design))
            # Pooled 8-family difference, its 95% interval and the non-inferiority verdict
            d = pooled(drawn[1])
            difference = float(np.mean([drawn[0][f][1][PRIMARY_METRIC] - drawn[0][f][0][PRIMARY_METRIC] for f in FAMILY_ORDER]))
            low, high = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            # Sensitivity: the same difference without the photos that fell back
            keep = set(test_photos[MANIFEST_PHOTO_COLUMN]) - fallback_photos
            primary.append({
                "arm": arm, "design": design, "metric": PRIMARY_METRIC, "margin": args.margin,
                "manual": float(np.mean([drawn[0][f][0][PRIMARY_METRIC] for f in FAMILY_ORDER])),
                "auto": float(np.mean([drawn[0][f][1][PRIMARY_METRIC] for f in FAMILY_ORDER])),
                "difference": difference, "ci_low": low, "ci_high": high,
                "p_noninferiority": float(np.mean(d <= -args.margin)), "verdict": verdict(low, high, args.margin),
                "fallback_photos": len(fallback_photos),
                "difference_without_fallback_photos": subset_difference(
                    pairs, design, keep, manual_union_map, auto_union_map),
                "resamples": int(len(d)),
            })

            # Secondary: models retrained on auto crops vs manual (only once retraining has finished)
            retrained_folders = {f: exp / f"auto_{arm}_{design}" / input_path / f for f in FAMILY_ORDER}
            if all((p / "test_predictions.csv").is_file() for p in retrained_folders.values()):
                again = family_set(retrained_folders, design, cache / arm / "retrained" / design)
                table = paired_comparison({f: (manual[design][f], again[f]) for f in FAMILY_ORDER},
                                          args.bootstrap, args.seed)
                retrained.append(tag(table.rename(columns={"full": "manual", "roi": "auto_retrained"}), arm=arm, design=design))
            # Secondary: auto ROI vs the full image
            table = paired_comparison({f: (full[f], auto[f]) for f in FAMILY_ORDER}, args.bootstrap, args.seed)
            versus_full.append(tag(table.rename(columns={"full": "full", "roi": "auto"}), arm=arm, design=design))

    # Holm over all primary tests, then write every table and arms.json
    primary_table = pd.DataFrame(primary)
    if not primary_table.empty:
        primary_table["p_holm"] = holm(primary_table["p_noninferiority"].tolist())
    primary_table.to_csv(out / "primary_noninferiority.csv", index=False)
    for name, parts in (("primary_by_family", by_family), ("secondary_retrained_vs_manual", retrained),
                        ("secondary_auto_vs_full", versus_full)):
        (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()).to_csv(out / f"{name}.csv", index=False)
    (out / "arms.json").write_text(json.dumps(arms, indent=2, default=float) + "\n", encoding="utf-8")

    # Log lines: dropped arms and each primary verdict
    for arm, info in arms.items():
        if not info["gate_passed"]:
            print(f"{arm}: failed its recall gate (recall {info['gate_recall']:.3f}); not compared")
    for row in primary:
        print(f"{row['arm']} {row['design']:<5} auto−manual {row['difference']:+.3f} [{row['ci_low']:+.3f}, "
              f"{row['ci_high']:+.3f}] → {row['verdict']} (margin {row['margin']})")
    print(f"Wrote primary_noninferiority.csv, primary_by_family.csv, secondary_*.csv and arms.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
