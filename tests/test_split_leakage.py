"""Unit and integration tests asserting zero case leakage and split integrity."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.create_case_split import (
    SPLIT_NAMES,
    DEFAULT_SPLIT_RATIOS,
    verify_leakage,
    target_group_counts,
)


def test_target_group_counts():
    counts = target_group_counts(421)
    assert counts[0] == 295  # 70% of 421 = 294.7 -> 295
    assert counts[1] == 63   # 15% of 421 = 63.15 -> 63
    assert counts[2] == 63   # remainder = 421 - 295 - 63 = 63
    assert sum(counts) == 421


def test_manifest_leakage_and_integrity():
    manifest_path = Path("data/splits/case_level_split_manifest.csv")
    assert manifest_path.exists(), "Manifest file must exist"

    df = pd.read_csv(manifest_path, dtype=str)
    assert len(df) == 1106, f"Expected 1106 rows, found {len(df)}"

    # 1. Check each case belongs to strictly ONE split
    case_splits = df.groupby("pseudonimized_id")["split"].nunique()
    assert (case_splits == 1).all(), "Case leakage! Cases assigned to multiple splits"

    # 2. Check pairwise disjointness of case sets
    train_cases = set(df[df["split"] == "train"]["pseudonimized_id"])
    val_cases = set(df[df["split"] == "validation"]["pseudonimized_id"])
    test_cases = set(df[df["split"] == "test"]["pseudonimized_id"])

    assert train_cases.isdisjoint(val_cases), "Train and Validation overlap!"
    assert train_cases.isdisjoint(test_cases), "Train and Test overlap!"
    assert val_cases.isdisjoint(test_cases), "Validation and Test overlap!"

    # 3. Check group counts
    assert len(train_cases) == 295
    assert len(val_cases) == 63
    assert len(test_cases) == 63
    assert len(train_cases | val_cases | test_cases) == 421

    # 4. Check all 7 canonical classes present in every split
    for split_name in SPLIT_NAMES:
        split_classes = df[df["split"] == split_name]["canonical_wound_label"].unique()
        assert len(split_classes) == 7, (
            f"Split '{split_name}' has only {len(split_classes)} classes, expected 7"
        )


def test_split_report_metadata():
    report_path = Path("data/splits/case_level_split_report.json")
    assert report_path.exists(), "Split report JSON must exist"

    with open(report_path) as f:
        report = json.load(f)

    assert report["seed"] == 42
    assert report["total_records"] == 1106
    assert report["total_unique_cases"] == 421
    assert report["case_counts_by_split"]["train"] == 295
    assert report["case_counts_by_split"]["validation"] == 63
    assert report["case_counts_by_split"]["test"] == 63
    assert report["leakage_check"] == "passed"
    assert report["frozen"] is True


if __name__ == "__main__":
    test_target_group_counts()
    test_manifest_leakage_and_integrity()
    test_split_report_metadata()
    print("All case-level split and leakage tests passed successfully!")
