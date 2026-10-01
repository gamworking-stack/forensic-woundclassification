"""Unit tests for metadata validation and label canonicalization."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.validate_metadata import (
    validate_metadata,
    CANONICAL_CLASSES,
    LABEL_NORMALIZATION_MAP,
)


def test_canonical_classes_count():
    assert len(CANONICAL_CLASSES) == 7
    expected = [
        "contusion",
        "cut wound",
        "graze wound",
        "gunshot wound",
        "laceration",
        "penetration wound",
        "scratch",
    ]
    assert sorted(CANONICAL_CLASSES) == sorted(expected)


def test_validate_metadata_synthetic_pass():
    df = pd.DataFrame(
        {
            "pseudonimized_id": [1, 1, 2, 3, 4, 5, 6, 7],
            "wound_id": [1, 2, 1, 1, 1, 1, 1, 1],
            "filename": [
                "a.jpg",
                "b.jpg",
                "c.jpg",
                "d.jpg",
                "e.jpg",
                "f.jpg",
                "g.jpg",
                "h.jpg",
            ],
            "wound_label": [
                "cut",
                "graze",
                "gsw",
                "contusion",
                "laceration",
                "scratch",
                "penetrating wound (sharp&blunt)",
                "cut wound",
            ],
        }
    )
    val_df, report = validate_metadata(df)
    assert report["status"] == "passed"
    assert len(report["errors"]) == 0
    assert report["total_records"] == 8
    assert report["total_unique_cases"] == 7
    assert report["canonical_classes_count"] == 7
    assert "canonical_wound_label" in val_df.columns
    assert "relative_path" in val_df.columns


def test_validate_metadata_missing_class():
    df = pd.DataFrame(
        {
            "pseudonimized_id": [1],
            "filename": ["a.jpg"],
            "wound_label": ["cut"],
        }
    )
    _, report = validate_metadata(df)
    assert report["status"] == "failed"
    assert any("missing required canonical class" in err.lower() for err in report["errors"])


def test_validate_metadata_actual_output():
    report_path = Path("data/metadata_validation_report.json")
    assert report_path.exists(), "metadata validation report must exist"

    with open(report_path) as f:
        report = json.load(f)

    assert report["status"] == "passed"
    assert report["errors"] == []
    assert report["total_records"] == 1106
    assert report["total_unique_cases"] == 421
    assert report["canonical_classes_count"] == 7


if __name__ == "__main__":
    test_canonical_classes_count()
    test_validate_metadata_synthetic_pass()
    test_validate_metadata_missing_class()
    test_validate_metadata_actual_output()
    print("All metadata validation unit tests passed!")
