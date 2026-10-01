#!/usr/bin/env python3
"""Audit the full-image dataset against the ROI dataset for a fair full vs ROI comparison.

Compares the authoritative case-level split manifest with the ROI manifest produced by
create_roi_dataset.py: which images were lost on the ROI path, image and case counts per
split and class for both paths, retention rates, and whether every ROI row still carries its
source image's case and split.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_roi_dataset import SPLIT_NAMES  # noqa: E402


JOIN_KEYS = ["canonical_wound_label", "filename"]
ALL = "all"


@dataclass
class AuditResult:
    classes: List[str] = field(default_factory=list)
    missing_from_roi: List[Dict[str, Any]] = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    coverage: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def retention_pct(kept: int, total: int) -> Optional[float]:
    return None if total == 0 else round(100.0 * kept / total, 2)


def build_audit_table(
    full_df: pd.DataFrame, roi_df: pd.DataFrame, classes: List[str], group_col: str
) -> pd.DataFrame:
    """Full vs ROI image and case counts for every split x class, plus ``all`` rollups.

    Case counts are per cell: in a class row, ``cases_lost`` counts cases that no longer have an
    ROI image of that class, even if the case survives through images of another class. Only
    the ``all`` class rows count cases lost from the ROI path entirely.
    """

    def subset(df: pd.DataFrame, split: str, label: str) -> pd.DataFrame:
        if split != ALL:
            df = df[df["split"] == split]
        if label != ALL:
            df = df[df["canonical_wound_label"] == label]
        return df

    records = []
    for split in [*SPLIT_NAMES, ALL]:
        for label in [*classes, ALL]:
            full, roi = subset(full_df, split, label), subset(roi_df, split, label)
            full_images, roi_images = len(full), len(roi)
            full_cases, roi_cases = full[group_col].nunique(), roi[group_col].nunique()
            records.append({
                "split": split,
                "canonical_wound_label": label,
                "full_images": full_images,
                "roi_images": roi_images,
                "images_lost": full_images - roi_images,
                "image_retention_pct": retention_pct(roi_images, full_images),
                "full_cases": full_cases,
                "roi_cases": roi_cases,
                "cases_lost": full_cases - roi_cases,
                "case_retention_pct": retention_pct(roi_cases, full_cases),
            })
    return pd.DataFrame(records)


def find_discrepancies(full_df: pd.DataFrame, roi_df: pd.DataFrame, group_col: str) -> List[Dict[str, Any]]:
    """ROI rows that are not in the full manifest or no longer carry their source case and split."""
    columns = [*JOIN_KEYS, group_col, "split"]
    joined = roi_df[columns].merge(
        full_df[columns], on=JOIN_KEYS, how="left", suffixes=("_roi", "_full"), indicator=True
    )
    discrepancies: List[Dict[str, Any]] = []
    for row in joined.to_dict("records"):
        base = {"filename": row["filename"], "canonical_wound_label": row["canonical_wound_label"]}
        if row["_merge"] == "left_only":
            discrepancies.append({**base, "type": "not_in_full_manifest"})
            continue
        if row[f"{group_col}_roi"] != row[f"{group_col}_full"]:
            discrepancies.append({**base, "type": "case_mismatch",
                                  "roi": row[f"{group_col}_roi"], "full": row[f"{group_col}_full"]})
        if row["split_roi"] != row["split_full"]:
            discrepancies.append({**base, "type": "split_mismatch",
                                  "roi": row["split_roi"], "full": row["split_full"]})
    return discrepancies


def path_coverage(df: pd.DataFrame, classes: List[str], group_col: str) -> Dict[str, Any]:
    """Case leakage and per-split class completeness for one input path."""
    splits_per_case = df.groupby(group_col)["split"].nunique()
    return {
        "leaked_cases": sorted(splits_per_case[splits_per_case > 1].index.tolist()),
        "missing_classes": {
            split: sorted(set(classes) - set(df.loc[df["split"] == split, "canonical_wound_label"]))
            for split in SPLIT_NAMES
        },
    }


def validate_manifest(df: pd.DataFrame, name: str, group_col: str) -> None:
    """Raise if a manifest lacks the audit columns or uses an unknown split name."""
    missing = {group_col, "split", *JOIN_KEYS}.difference(df.columns)
    if missing:
        raise ValueError(f"{name} manifest missing columns: {sorted(missing)}")
    unknown = sorted(set(df["split"]).difference(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"{name} manifest has unknown split values: {unknown}")


def audit_full_vs_roi(
    full_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    group_col: str = "pseudonimized_id",
    drop_reasons: Optional[Dict[Tuple[str, str], str]] = None,
) -> AuditResult:
    """Compare the full-image manifest with the ROI manifest.

    ``drop_reasons`` maps (canonical label, filename) to the reason create_roi_dataset.py gave
    for not cropping that image; missing images without one are marked ``unexplained``.
    """
    validate_manifest(full_df, "full", group_col)
    validate_manifest(roi_df, "ROI", group_col)
    drop_reasons = drop_reasons or {}

    joined = full_df.merge(roi_df[JOIN_KEYS], on=JOIN_KEYS, how="left", indicator=True)
    missing = joined[joined["_merge"] == "left_only"]
    columns = [group_col, "filename", "canonical_wound_label", "split"]
    roi_cases = set(roi_df[group_col])
    missing_records = [
        {
            **record,
            "reason": drop_reasons.get((record["canonical_wound_label"], record["filename"]), "unexplained"),
            "case_lost_entirely": record[group_col] not in roi_cases,
        }
        for record in missing[columns].to_dict("records")
    ]
    classes = sorted(full_df["canonical_wound_label"].unique())
    return AuditResult(
        classes=classes,
        missing_from_roi=missing_records,
        table=build_audit_table(full_df, roi_df, classes, group_col),
        discrepancies=find_discrepancies(full_df, roi_df, group_col),
        coverage={
            "full": path_coverage(full_df, classes, group_col),
            "roi": path_coverage(roi_df, classes, group_col),
        },
    )


def load_drop_reasons(crop_summary: Path) -> Dict[Tuple[str, str], str]:
    """Read per-image drop reasons from create_roi_dataset.py's crop_summary.json."""
    with crop_summary.open(encoding="utf-8") as f:
        summary = json.load(f)
    dropped = summary.get("unannotated_records", []) + summary.get("skipped_records", [])
    return {(d["canonical_wound_label"], d["filename"]): d["reason"] for d in dropped}


def audit_problems(audit: AuditResult) -> List[str]:
    """Integrity failures that make the full vs ROI comparison unfair."""
    problems = []
    if audit.discrepancies:
        problems.append(f"{len(audit.discrepancies)} ROI rows break split/case inheritance")
    unexplained = [m["filename"] for m in audit.missing_from_roi if m["reason"] == "unexplained"]
    if unexplained:
        problems.append(f"images missing from ROI with no recorded drop reason: {unexplained}")
    for path_name, cov in audit.coverage.items():
        if cov["leaked_cases"]:
            problems.append(f"{path_name}: cases in more than one split: {cov['leaked_cases']}")
        for split, missing in cov["missing_classes"].items():
            if missing:
                problems.append(f"{path_name}: {split} is missing classes {missing}")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the full-image dataset against the ROI dataset.")
    parser.add_argument(
        "--full-manifest",
        type=Path,
        default=Path("data/splits/case_level_split_manifest.csv"),
        help="Authoritative case-level split manifest (full-image path).",
    )
    parser.add_argument(
        "--roi-manifest",
        type=Path,
        default=Path("data/roi/roi_manifest.csv"),
        help="ROI manifest produced by create_roi_dataset.py.",
    )
    parser.add_argument(
        "--crop-summary",
        type=Path,
        default=Path("data/roi/crop_summary.json"),
        help="crop_summary.json from create_roi_dataset.py, used to explain why images were dropped.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audits"),
        help="Directory to save full_vs_roi_audit_table.csv and full_vs_roi_audit_report.json.",
    )
    parser.add_argument(
        "--group-column",
        default="pseudonimized_id",
        help="Grouping column for cases.",
    )
    return parser.parse_args()


def table_row(table: pd.DataFrame, split: str, label: str) -> Dict[str, Any]:
    match = table[(table["split"] == split) & (table["canonical_wound_label"] == label)]
    return match.iloc[0].to_dict()


def main() -> int:
    args = parse_args()
    group_col = args.group_column
    for path in (args.full_manifest, args.roi_manifest, args.crop_summary):
        if not path.exists():
            print(f"Error: input not found: {path}", file=sys.stderr)
            return 1

    full_df = pd.read_csv(args.full_manifest, dtype=str)
    roi_df = pd.read_csv(args.roi_manifest, dtype=str)
    audit = audit_full_vs_roi(full_df, roi_df, group_col, drop_reasons=load_drop_reasons(args.crop_summary))
    problems = audit_problems(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_out = args.output_dir / "full_vs_roi_audit_table.csv"
    audit.table.to_csv(table_out, index=False)

    overall = table_row(audit.table, ALL, ALL)
    report = {
        "full_manifest": str(args.full_manifest),
        "roi_manifest": str(args.roi_manifest),
        "crop_summary": str(args.crop_summary),
        "audit_passed": not problems,
        "problems": problems,
        "totals": {
            "full_images": int(overall["full_images"]),
            "roi_images": int(overall["roi_images"]),
            "images_lost": int(overall["images_lost"]),
            "full_cases": int(overall["full_cases"]),
            "roi_cases": int(overall["roi_cases"]),
            "cases_lost": int(overall["cases_lost"]),
        },
        "missing_from_roi_count": len(audit.missing_from_roi),
        "missing_from_roi": audit.missing_from_roi,
        "retention_by_class": {
            label: {
                "image_retention_pct": table_row(audit.table, ALL, label)["image_retention_pct"],
                "case_retention_pct": table_row(audit.table, ALL, label)["case_retention_pct"],
            }
            for label in audit.classes
        },
        "images_by_split": {
            split: {
                "full": int(table_row(audit.table, split, ALL)["full_images"]),
                "roi": int(table_row(audit.table, split, ALL)["roi_images"]),
            }
            for split in SPLIT_NAMES
        },
        "coverage": audit.coverage,
        "discrepancies": audit.discrepancies,
        "table_notes": (
            "Per-class cases_lost counts cases with no ROI image of that class; such a case may "
            "survive through other classes. The all-class rows count cases lost entirely."
        ),
    }
    report_out = args.output_dir / "full_vs_roi_audit_report.json"
    with report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    totals = report["totals"]
    print("--- Full vs ROI Audit ---")
    print(f"Images: full {totals['full_images']:,} -> ROI {totals['roi_images']:,} ({totals['images_lost']} lost)")
    print(f"Cases:  full {totals['full_cases']:,} -> ROI {totals['roi_cases']:,} ({totals['cases_lost']} lost)")
    print("\nImages missing from ROI:")
    for m in audit.missing_from_roi:
        lost = "  [whole case lost]" if m["case_lost_entirely"] else ""
        print(f"  - case {m[group_col]:>4}: {m['filename']:<16} {m['canonical_wound_label']:<18} "
              f"{m['split']:<11} {m['reason']}{lost}")
    print("\nImage retention by class:")
    for label, rates in report["retention_by_class"].items():
        print(f"  - {label:<20}: {rates['image_retention_pct']:>6}%")
    print(f"\nTable saved:  {table_out}")
    print(f"Report saved: {report_out}")

    if problems:
        for problem in problems:
            print(f"Error: {problem}", file=sys.stderr)
        return 1
    print("\nAudit passed: no discrepancies, no case leakage, all classes in every split for both paths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
