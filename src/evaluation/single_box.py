#!/usr/bin/env python3
"""Single-box test photos (T21): one full image against one crop of the same wound.

On a test photo with exactly one annotated box, the photo, its per-box crop and its union crop
show the same single wound, so a full-image model and a crop model each make one prediction from
one input and nothing is averaged. This script keeps those photos in the evaluated test files of
the main run (full image, union ROI) and the per-crop run (S25-S32), and compares full image with
per-crop ROI and with union ROI using the T15 paired case-level bootstrap. Post hoc: decided after
every test result had been seen (plan: tickets/T21_single_box.md §2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_roi_dataset import IMAGE_EXTENSIONS  # noqa: E402
from src.evaluation.evaluate import CROP_PATH, CROP_SCENARIO_IDS, FAMILY_ORDER, SCENARIO_IDS  # noqa: E402
from src.evaluation.metrics import Predictions, load_predictions, summary_metrics  # noqa: E402
from src.evaluation.plots import plot_full_vs_roi  # noqa: E402
from src.evaluation.summarize import POOLED, paired_comparison, scenario_folder  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402


PATHS = ("full", "roi", CROP_PATH)  # full image and union ROI from the main run, per-crop from T18
TABLE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")
PHOTO = "photo"  # added column: the manifest's relative_path of each row's photo


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full image vs one crop on the test photos with a single box.")
    parser.add_argument("--main-root", type=Path, required=True,
                        help="Main run (full/ and roi/ for all 8 families), already evaluated.")
    parser.add_argument("--crop-root", type=Path, required=True,
                        help="Per-crop run laid out as crop/<family>/, already evaluated.")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--roi-manifest", type=Path, default=DATA_DIR / "roi/roi_manifest.csv")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Paired case-level resamples.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def photo_stem(name: str) -> str:
    # File name without folder or image extension; manifest names may lack the extension
    path = Path(str(name).strip())
    return path.stem if path.suffix.lower() in IMAGE_EXTENSIONS else path.name


def single_box_photos(box_manifest: Path) -> pd.DataFrame:
    """Test rows of the per-box manifest whose photo has exactly one box, one row per photo."""
    rows = pd.read_csv(box_manifest, dtype=str, keep_default_na=False)
    test = rows[rows["split"] == "test"]
    boxes = test.groupby("relative_path")["relative_path"].transform("size")
    return test[boxes == 1].reset_index(drop=True)


def photo_lookup(photos: pd.DataFrame, roi_manifest: Path) -> Dict[str, Dict]:
    """Per input path, the key a prediction row is matched on and the photo it belongs to."""
    # Full images: the prediction file names the raw photo, so match case and file stem
    full = {(r["pseudonimized_id"], photo_stem(r["filename"])): r["relative_path"] for r in photos.to_dict("records")}
    if len(full) != len(photos):
        raise ValueError("Two single-box test photos share a case and file stem; full-image matching is ambiguous")
    # Union ROI: the union crop of the same photo, from the union manifest
    union = pd.read_csv(roi_manifest, dtype=str, keep_default_na=False)
    union = union[union["relative_path"].isin(photos["relative_path"])]
    # Per-crop: the photo's only per-box crop
    return {"full": full,
            "roi": dict(zip(union["roi_relative_path"], union["relative_path"])),
            CROP_PATH: dict(zip(photos["roi_relative_path"], photos["relative_path"]))}


def row_key(frame: pd.DataFrame, input_path: str) -> pd.Series:
    if input_path == "full":
        return pd.Series(list(zip(frame["pseudonimized_id"], frame["relative_path"].map(photo_stem))), index=frame.index)
    return frame["relative_path"]


def restrict(path: Path, input_path: str, lookup: Dict, photos: pd.DataFrame) -> pd.DataFrame:
    """The prediction rows of the single-box photos, each photo exactly once, named in PHOTO."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame[PHOTO] = row_key(frame, input_path).map(lambda k: lookup.get(k))
    kept = frame[frame[PHOTO].notna()]
    # Every single-box photo must be found once, under its own case and class
    found = kept[PHOTO].value_counts()
    missing = sorted(set(photos["relative_path"]) - set(found.index))
    if missing or (found > 1).any():
        raise ValueError(f"{path}: {len(missing)} single-box photos missing, "
                         f"{int((found > 1).sum())} matched more than once (e.g. {(missing or list(found.index))[:3]})")
    expected = photos.set_index("relative_path").loc[kept[PHOTO]]
    if (expected["pseudonimized_id"].to_numpy() != kept["pseudonimized_id"].to_numpy()).any() or \
            (expected["canonical_wound_label"].to_numpy() != kept["true_class"].to_numpy()).any():
        raise ValueError(f"{path}: a matched row has another case or class than its single-box photo")
    return kept.sort_values(PHOTO)


def scenario_id(input_path: str, family: str) -> str:
    return CROP_SCENARIO_IDS[(CROP_PATH, family)] if input_path == CROP_PATH else SCENARIO_IDS[(input_path, family)]


def subset_predictions(args: argparse.Namespace, photos: pd.DataFrame, out: Path
                       ) -> Dict[Tuple[str, str], Predictions]:
    """Write and load the single-box test predictions of every family and input path."""
    lookup = photo_lookup(photos, args.roi_manifest)
    roots = {"full": args.main_root, "roi": args.main_root, CROP_PATH: args.crop_root}
    predictions = {}
    for input_path in PATHS:
        for family in FAMILY_ORDER:
            # Keep the single-box rows; the per-image file stays outside summary/, like every prediction file
            source = scenario_folder(roots[input_path], input_path, family) / "test_predictions.csv"
            target = out / input_path / family / "test_predictions.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            restrict(source, input_path, lookup[input_path], photos).to_csv(target, index=False)
            predictions[(input_path, family)] = load_predictions(target, "test")
    return predictions


def metrics_table(predictions: Dict[Tuple[str, str], Predictions]) -> pd.DataFrame:
    # One row per scenario with its single-box test size and point metrics
    rows = []
    for (input_path, family), p in predictions.items():
        m = summary_metrics(p.y_true, p.y_pred, p.probs)
        rows.append({"scenario_id": scenario_id(input_path, family), "family": family, "input_path": input_path,
                     "test_photos": len(p.y_true), "test_cases": len(set(p.cases)),
                     **{name: m[name] for name in TABLE_METRICS}})
    return pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)


def class_counts(photos: pd.DataFrame) -> pd.DataFrame:
    # Photos and cases per class, the size behind every single-box metric
    counts = photos.groupby("canonical_wound_label").agg(photos=("relative_path", "size"),
                                                         cases=("pseudonimized_id", "nunique"))
    total = pd.DataFrame({"photos": [len(photos)], "cases": [photos["pseudonimized_id"].nunique()]}, index=["all"])
    return pd.concat([counts, total]).rename_axis("class").reset_index()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Subset prediction files go to <crop-root>/single_box, tables and plots to <crop-root>/summary
    out = args.crop_root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    photos = single_box_photos(args.box_manifest)
    predictions = subset_predictions(args, photos, args.crop_root / "single_box")
    class_counts(photos).to_csv(out / "single_box_counts.csv", index=False)
    metrics_table(predictions).to_csv(out / "single_box_metrics.csv", index=False)

    # Per-crop and union ROI each against the full image, on the same single-box photos
    for name, label in ((CROP_PATH, "per-crop ROI"), ("roi", "union ROI")):
        pairs = {f: (predictions[("full", f)], predictions[(name, f)]) for f in FAMILY_ORDER}
        table = paired_comparison(pairs, args.bootstrap, args.seed)
        table.to_csv(out / f"single_box_{name}_vs_full.csv", index=False)
        plot_full_vs_roi(table[table["metric"] == "macro_f1"], POOLED,
                         f"Single-box test photos, macro-F1: {label} minus full image",
                         out / f"single_box_{name}_vs_full_macro_f1.png", labels=("full image", label))
        # Print the macro-F1 differences
        for row in table[table["metric"] == "macro_f1"].to_dict("records"):
            holm = "" if pd.isna(row["p_holm"]) else f" holm={row['p_holm']:.3f}"
            print(f"{row['family']:<22} {name}-full {row['difference']:+.3f} [{row['ci_low']:+.3f}, "
                  f"{row['ci_high']:+.3f}] p={row['p_value']:.3f}{holm}", flush=True)
    print(f"{len(photos)} single-box test photos from {photos['pseudonimized_id'].nunique()} cases; "
          f"wrote single_box_*.csv and *.png to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
