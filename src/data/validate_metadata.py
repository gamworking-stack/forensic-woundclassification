#!/usr/bin/env python3
"""Validate wound metadata CSV/Excel and normalize labels to canonical classes.

Checks data completeness, validates required schema, detects duplicate
filenames or cross-case mappings, maps raw wound labels to the 7 canonical
classes, and outputs a validated CSV along with a comprehensive JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# Canonical 7 wound classes defined in the thesis proposal
CANONICAL_CLASSES = [
    "contusion",
    "cut wound",
    "graze wound",
    "gunshot wound",
    "laceration",
    "penetration wound",
    "scratch",
]

# Mapping dictionary for known spelling / formatting variations
LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    # Cut wound
    "cut": "cut wound",
    "cut wound": "cut wound",
    "cuts": "cut wound",
    # Graze wound
    "graze": "graze wound",
    "graze wound": "graze wound",
    "grazes": "graze wound",
    # Gunshot wound
    "gsw": "gunshot wound",
    "gunshot wound": "gunshot wound",
    "gunshot": "gunshot wound",
    # Contusion
    "contusion": "contusion",
    "contusions": "contusion",
    "bruise": "contusion",
    # Laceration
    "laceration": "laceration",
    "lacerations": "laceration",
    # Scratch
    "scratch": "scratch",
    "scratches": "scratch",
    # Penetration wound
    "penetrating wound (sharp&blunt)": "penetration wound",
    "penetration wound": "penetration wound",
    "penetrating wound": "penetration wound",
    "penetration": "penetration wound",
    "penetrating": "penetration wound",
    "stab": "penetration wound",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate forensic wound classification metadata and canonicalize labels."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("../thesis_data_30_sep_id_only.csv"),
        help="Path to source metadata CSV or Excel file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/validated_metadata.csv"),
        help="Destination path for validated metadata CSV.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/metadata_validation_report.json"),
        help="Destination path for validation summary JSON.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory containing raw image folders to check physical file existence.",
    )
    return parser.parse_args()


def load_metadata(file_path: Path) -> pd.DataFrame:
    """Load metadata from CSV or Excel file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input metadata file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        # If excel, prefer 'modified' or first sheet
        xl = pd.ExcelFile(file_path)
        sheet_to_load = "modified" if "modified" in xl.sheet_names else 0
        df = pd.read_excel(file_path, sheet_name=sheet_to_load)
    else:
        df = pd.read_csv(file_path)

    # Strip any leading/trailing whitespace from string columns and column names
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def validate_metadata(
    df: pd.DataFrame,
    image_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate schema, completeness, labels, and filenames.

    Returns the validated dataframe (with canonical columns added) and
    the validation report dictionary.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Required columns check
    required_cols = ["pseudonimized_id", "filename", "wound_label"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        return df, {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
        }

    # Clean working copy
    df = df.copy()

    # Drop completely empty rows if any
    df = df.dropna(subset=required_cols, how="all")

    # 2. Check for null or empty values in required fields
    for col in required_cols:
        null_count = int(df[col].isnull().sum())
        empty_str_count = int((df[col].astype(str).str.strip() == "").sum())
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null/NaN values.")
        if empty_str_count > 0:
            errors.append(f"Column '{col}' has {empty_str_count} empty string values.")

    # 3. Label Canonicalization
    def map_label(val: Any) -> Optional[str]:
        raw_clean = str(val).strip().lower()
        return LABEL_NORMALIZATION_MAP.get(raw_clean, None)

    df["canonical_wound_label"] = df["wound_label"].apply(map_label)
    unmapped = df[df["canonical_wound_label"].isnull()]["wound_label"].unique().tolist()
    if unmapped:
        errors.append(f"Encountered unknown wound label(s) that could not be mapped: {unmapped}")

    # Check presence of all 7 canonical classes
    present_classes = sorted(df["canonical_wound_label"].dropna().unique().tolist())
    missing_classes = [c for c in CANONICAL_CLASSES if c not in present_classes]
    if missing_classes:
        errors.append(f"Dataset is missing required canonical class(es): {missing_classes}")

    # 4. Filename integrity and extensions
    missing_ext_mask = ~df["filename"].astype(str).str.contains(r"\.[a-zA-Z0-9]+$", regex=True)
    missing_ext_count = int(missing_ext_mask.sum())
    if missing_ext_count > 0:
        sample_missing = df[missing_ext_mask]["filename"].head(5).tolist()
        warnings.append(
            f"{missing_ext_count} filenames lack a file extension (e.g. {sample_missing})."
        )

    # 5. Duplicate filename analysis
    dup_mask = df["filename"].duplicated(keep=False)
    dup_count = int(df["filename"].duplicated().sum())
    dup_details: List[Dict[str, Any]] = []
    if dup_count > 0:
        dup_filenames = df[df["filename"].duplicated()]["filename"].unique().tolist()
        for fname in dup_filenames:
            instances = df[df["filename"] == fname][
                ["pseudonimized_id", "wound_id" if "wound_id" in df.columns else "filename", "wound_label"]
            ].to_dict(orient="records")
            dup_details.append({"filename": fname, "instances": instances})

        warnings.append(
            f"Found {dup_count} duplicate filename occurrences across rows ({len(dup_filenames)} unique filenames). "
            f"Some images depict multiple wounds or share filenames."
        )

    # Construct relative path convention: <canonical_wound_label>/<filename>
    df["relative_path"] = df["canonical_wound_label"] + "/" + df["filename"]

    # 6. Physical image existence check (optional)
    image_existence_checked = False
    missing_images_count = 0
    if image_root is not None:
        image_root = Path(image_root)
        if image_root.exists():
            image_existence_checked = True
            missing_paths = []
            for _, row in df.iterrows():
                fname = str(row["filename"])
                label_dir = row["canonical_wound_label"]
                # Try direct path, extension variations
                candidates = [
                    image_root / label_dir / fname,
                    image_root / str(row["wound_label"]) / fname,
                    image_root / fname,
                ]
                if not any(c.exists() for c in candidates):
                    missing_paths.append(fname)
            missing_images_count = len(missing_paths)
            if missing_images_count > 0:
                warnings.append(
                    f"{missing_images_count} images were not found under image-root: '{image_root}'"
                )
        else:
            warnings.append(f"Specified image-root does not exist: '{image_root}'")

    # 7. Compute dynamic statistics (Zero hardcoded counts)
    total_images = len(df)
    unique_cases = int(df["pseudonimized_id"].nunique())
    per_class_image_counts = (
        df["canonical_wound_label"].value_counts().to_dict()
    )
    per_class_case_counts = (
        df.groupby("canonical_wound_label")["pseudonimized_id"].nunique().to_dict()
    )

    # Multi-wound case stats
    case_wound_counts = df.groupby("pseudonimized_id").size()
    multi_image_cases = int((case_wound_counts > 1).sum())

    case_label_counts = df.groupby("pseudonimized_id")["canonical_wound_label"].nunique()
    multi_label_cases = int((case_label_counts > 1).sum())

    status = "passed" if len(errors) == 0 else "failed"

    report: Dict[str, Any] = {
        "status": status,
        "total_records": total_images,
        "total_unique_cases": unique_cases,
        "canonical_classes_count": len(present_classes),
        "canonical_classes": present_classes,
        "per_class_image_counts": per_class_image_counts,
        "per_class_case_counts": per_class_case_counts,
        "multi_image_cases_count": multi_image_cases,
        "multi_label_cases_count": multi_label_cases,
        "missing_extensions_count": missing_ext_count,
        "duplicate_filenames": dup_details,
        "image_root_checked": image_existence_checked,
        "missing_images_count": missing_images_count,
        "warnings": warnings,
        "errors": errors,
    }

    return df, report


def main() -> int:
    args = parse_args()

    print(f"Loading metadata from: {args.input_csv}")
    try:
        df = load_metadata(args.input_csv)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        return 1

    print(f"Validating {len(df)} records...")
    validated_df, report = validate_metadata(df, image_root=args.image_root)

    # Ensure output directories exist
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved validation report to: {args.output_report}")

    if report["status"] != "passed":
        print("Validation FAILED with the following errors:", file=sys.stderr)
        for err in report["errors"]:
            print(f"  - {err}", file=sys.stderr)
        return 1

    # Save validated CSV
    validated_df.to_csv(args.output_csv, index=False)
    print(f"Saved validated metadata ({len(validated_df)} rows) to: {args.output_csv}")

    print("\n--- Validation Summary ---")
    print(f"Total Records: {report['total_records']}")
    print(f"Total Cases:   {report['total_unique_cases']}")
    print(f"All 7 Canonical Classes Present: {report['canonical_classes_count'] == 7}")
    print("\nPer-class Image Counts:")
    for cls, cnt in report["per_class_image_counts"].items():
        case_cnt = report["per_class_case_counts"].get(cls, 0)
        print(f"  - {cls:<20}: {cnt:>4} images ({case_cnt:>3} cases)")

    if report["warnings"]:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for warn in report["warnings"]:
            print(f"  * {warn}")

    print("\nValidation PASSED successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
