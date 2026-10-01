#!/usr/bin/env python3
"""Compare a live-mode ROI manifest against the committed dry-run manifest.

The committed `data/roi/roi_manifest.csv` was built in dry-run mode, with image
dimensions taken from a reference manifest. A live run on the HPC reads the real
images, so its crop boxes must match the dry run exactly. Any difference in case,
split, image dimensions or box coordinates is reported, and the script exits non-zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


KEY_COLUMN = "roi_relative_path"
COMPARED_COLUMNS = (
    "pseudonimized_id",
    "split",
    "canonical_wound_label",
    "bbox_count",
    "x1",
    "y1",
    "x2",
    "y2",
    "source_width",
    "source_height",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a live ROI run against the dry-run manifest.")
    parser.add_argument("--expected", type=Path, default=Path("data/roi/roi_manifest.csv"),
                        help="Committed dry-run ROI manifest.")
    parser.add_argument("--actual", type=Path, required=True,
                        help="ROI manifest written by the live run.")
    parser.add_argument("--crop-root", type=Path, default=None,
                        help="Live output directory; when given, every crop file must exist there.")
    return parser.parse_args()


def compare(expected: pd.DataFrame, actual: pd.DataFrame) -> list[str]:
    """Return human-readable differences between two ROI manifests (empty when identical)."""
    problems: list[str] = []
    for name, df in (("expected", expected), ("actual", actual)):
        dupes = df[KEY_COLUMN][df[KEY_COLUMN].duplicated()].tolist()
        if dupes:
            problems.append(f"{name} manifest has duplicate {KEY_COLUMN}: {dupes[:5]}")
    if problems:
        return problems

    exp = expected.set_index(KEY_COLUMN)
    act = actual.set_index(KEY_COLUMN)
    missing = sorted(set(exp.index) - set(act.index))
    extra = sorted(set(act.index) - set(exp.index))
    if missing:
        problems.append(f"{len(missing)} crops missing from live run, e.g. {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} crops only in live run, e.g. {extra[:5]}")

    shared = sorted(set(exp.index) & set(act.index))
    for col in COMPARED_COLUMNS:
        if col not in exp.columns or col not in act.columns:
            problems.append(f"column {col!r} missing from a manifest")
            continue
        diff = [k for k in shared if exp.at[k, col] != act.at[k, col]]
        if diff:
            k = diff[0]
            problems.append(
                f"{len(diff)} rows differ in {col!r}, e.g. {k}: expected {exp.at[k, col]!r}, got {act.at[k, col]!r}"
            )
    return problems


def main() -> int:
    args = parse_args()
    expected = pd.read_csv(args.expected, dtype=str, keep_default_na=False)
    actual = pd.read_csv(args.actual, dtype=str, keep_default_na=False)

    problems = compare(expected, actual)
    if args.crop_root is not None:
        absent = [p for p in actual[KEY_COLUMN] if not (args.crop_root / p).is_file()]
        if absent:
            problems.append(f"{len(absent)} crop files not on disk under {args.crop_root}, e.g. {absent[:5]}")

    print(f"Expected rows: {len(expected):,}  Actual rows: {len(actual):,}")
    if problems:
        print("ROI comparison FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("ROI comparison PASSED: live crops match the committed dry-run manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
