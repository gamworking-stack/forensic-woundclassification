#!/usr/bin/env python3
"""Create a reproducible 70/15/15 case-level grouped split for forensic wound classification.

All rows sharing the same grouping identifier (`pseudonimized_id`) are assigned
strictly to one split partition: train, validation, or test.
A seeded multi-objective search optimizes class balance across partitions despite
cases having multi-wound occurrences, guaranteeing every class appears in all splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = np.array([0.70, 0.15, 0.15], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition wound metadata by case identifier (pseudonimized_id) without leakage."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/validated_metadata.csv"),
        help="Path to validated metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory to save generated split manifests and reports.",
    )
    parser.add_argument(
        "--group-column",
        default="pseudonimized_id",
        help="Grouping column for cases/patients (no case will cross partitions).",
    )
    parser.add_argument(
        "--label-column",
        default="canonical_wound_label",
        help="Column identifying wound class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for search reproducibility.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50_000,
        help="Number of candidate grouped assignments to evaluate.",
    )
    parser.add_argument(
        "--expected-classes",
        type=int,
        default=7,
        help="Expected number of unique wound classes.",
    )
    return parser.parse_args()


def target_group_counts(number_of_groups: int, ratios: np.ndarray = DEFAULT_SPLIT_RATIOS) -> np.ndarray:
    """Calculate target number of cases in train, val, and test partitions."""
    train_count = int(round(number_of_groups * ratios[0]))
    val_count = int(round(number_of_groups * ratios[1]))
    test_count = number_of_groups - train_count - val_count
    return np.array([train_count, val_count, test_count], dtype=int)


def load_dataset(input_csv: Path, group_col: str, label_col: str, expected_classes: int) -> pd.DataFrame:
    """Load metadata and validate basic prerequisites for splitting."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    # If label_col not in df but fallback available
    if label_col not in df.columns:
        if "wound_label" in df.columns:
            label_col = "wound_label"
        else:
            raise KeyError(f"Label column '{label_col}' not found in {input_csv}")

    for col in (group_col, label_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {input_csv}")
        df[col] = df[col].astype(str).str.strip()

    labels = sorted(df[label_col].unique())
    if len(labels) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, but found {len(labels)}: {labels}")

    return df


def build_group_matrices(
    df: pd.DataFrame, group_col: str, label_col: str
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Build group-by-class contingency matrices for images and case presence."""
    group_ids = np.array(sorted(df[group_col].unique()), dtype=object)
    labels = sorted(df[label_col].unique())

    image_counts = (
        pd.crosstab(df[group_col], df[label_col])
        .reindex(index=group_ids, columns=labels, fill_value=0)
    )
    image_matrix = image_counts.to_numpy(dtype=float)
    presence_matrix = (image_matrix > 0).astype(float)
    return group_ids, labels, image_matrix, presence_matrix


def candidate_score(
    assignment: np.ndarray,
    image_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    ratios: np.ndarray = DEFAULT_SPLIT_RATIOS,
) -> float:
    """Evaluate discrepancy between candidate split distributions and ideal ratios."""
    image_totals = image_matrix.sum(axis=0)
    presence_totals = presence_matrix.sum(axis=0)

    expected_images = ratios[:, None] * image_totals[None, :]
    expected_presence = ratios[:, None] * presence_totals[None, :]

    observed_images = np.vstack(
        [image_matrix[assignment == i].sum(axis=0) for i in range(3)]
    )
    observed_presence = np.vstack(
        [presence_matrix[assignment == i].sum(axis=0) for i in range(3)]
    )

    # Hard constraint: every class must have at least 1 image and case in each partition
    if (observed_images == 0).any() or (observed_presence == 0).any():
        return float("inf")

    # Weighted quadratic penalty on deviations
    image_error = np.square(observed_images - expected_images) / (expected_images + 1.0)
    presence_error = np.square(observed_presence - expected_presence) / (expected_presence + 1.0)
    return float(image_error.sum() + presence_error.sum())


def find_optimal_split(
    number_of_groups: int,
    image_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    seed: int = 42,
    trials: int = 50_000,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Search candidate random group assignments and pick the one with lowest loss."""
    counts = target_group_counts(number_of_groups)
    rng = np.random.default_rng(seed)
    best_assignment: np.ndarray | None = None
    best_score = float("inf")

    train_end = counts[0]
    val_end = train_end + counts[1]

    for _ in range(trials):
        order = rng.permutation(number_of_groups)
        assignment = np.empty(number_of_groups, dtype=np.int8)
        assignment[order[:train_end]] = 0
        assignment[order[train_end:val_end]] = 1
        assignment[order[val_end:]] = 2

        score = candidate_score(assignment, image_matrix, presence_matrix)
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None or not np.isfinite(best_score):
        raise RuntimeError(
            "No candidate split placed all 7 classes in every partition. "
            "Increase --trials or verify class representation."
        )

    return best_assignment, best_score, counts


def verify_leakage(df: pd.DataFrame, group_col: str, expected_group_counts: np.ndarray) -> None:
    """Assert zero leakage across train, val, and test subsets."""
    splits_per_group = df.groupby(group_col)["split"].nunique()
    if not splits_per_group.eq(1).all():
        leaked_groups = splits_per_group[splits_per_group > 1].index.tolist()
        raise AssertionError(f"Case leakage detected! Groups spanning multiple splits: {leaked_groups}")

    group_sets = {
        name: set(df.loc[df["split"] == name, group_col])
        for name in SPLIT_NAMES
    }

    assert group_sets["train"].isdisjoint(group_sets["validation"]), "Leakage between train and validation!"
    assert group_sets["train"].isdisjoint(group_sets["test"]), "Leakage between train and test!"
    assert group_sets["validation"].isdisjoint(group_sets["test"]), "Leakage between validation and test!"

    observed = np.array([len(group_sets[name]) for name in SPLIT_NAMES])
    if not np.array_equal(observed, expected_group_counts):
        raise AssertionError(f"Group count mismatch: observed={observed}, expected={expected_group_counts}")


def generate_summary(df: pd.DataFrame, group_col: str, label_col: str) -> pd.DataFrame:
    """Generate detailed breakdown of image counts and unique cases per split and class."""
    img_counts = df.groupby(["split", label_col], observed=True).size().rename("image_count")
    grp_counts = df.groupby(["split", label_col], observed=True)[group_col].nunique().rename("case_count")

    summary = pd.concat([img_counts, grp_counts], axis=1).reset_index()
    summary["image_pct_in_class"] = summary.groupby(label_col)["image_count"].transform(
        lambda s: s / s.sum() * 100.0
    ).round(2)
    summary["case_pct_in_class"] = summary.groupby(label_col)["case_count"].transform(
        lambda s: s / s.sum() * 100.0
    ).round(2)
    return summary.sort_values([label_col, "split"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    print(f"Loading validated metadata: {args.input_csv}")
    df = load_dataset(args.input_csv, args.group_column, args.label_column, args.expected_classes)

    group_ids, labels, image_matrix, presence_matrix = build_group_matrices(
        df, args.group_column, args.label_column
    )

    print(
        f"Optimizing grouped split across {len(group_ids)} cases, {len(labels)} classes "
        f"(seed={args.seed}, trials={args.trials:,})..."
    )
    assignment, score, group_counts = find_optimal_split(
        len(group_ids), image_matrix, presence_matrix, seed=args.seed, trials=args.trials
    )

    group_df = pd.DataFrame({
        args.group_column: group_ids,
        "split": [SPLIT_NAMES[idx] for idx in assignment],
    })

    # Merge partition assignment back into metadata dataframe
    split_df = df.merge(group_df, on=args.group_column, how="left", validate="many_to_one")

    # Verify zero leakage
    verify_leakage(split_df, args.group_column, group_counts)

    # Generate summary table
    summary_df = generate_summary(split_df, args.group_column, args.label_column)

    # Save artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "case_level_split_manifest.csv"
    assignment_path = args.output_dir / "case_split_assignments.csv"
    summary_path = args.output_dir / "case_level_split_summary.csv"
    report_path = args.output_dir / "case_level_split_report.json"

    split_df.to_csv(manifest_path, index=False)
    group_df.to_csv(assignment_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    split_img_counts = split_df["split"].value_counts().to_dict()

    report: Dict[str, Any] = {
        "input_csv": str(args.input_csv.resolve()),
        "group_column": args.group_column,
        "label_column": args.label_column,
        "seed": args.seed,
        "trials": args.trials,
        "optimization_score": score,
        "canonical_classes": labels,
        "total_records": len(split_df),
        "total_unique_cases": len(group_ids),
        "case_counts_by_split": {
            name: int(cnt) for name, cnt in zip(SPLIT_NAMES, group_counts)
        },
        "image_counts_by_split": {
            name: int(split_img_counts.get(name, 0)) for name in SPLIT_NAMES
        },
        "leakage_check": "passed",
        "frozen": True,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n--- Split Generation Summary ---")
    print(f"Total Rows:     {len(split_df):,}")
    print(f"Total Cases:    {len(group_ids):,}")
    print(f"Optimal Score:  {score:.6f}")
    print(f"Case Allocation: Train={group_counts[0]}, Validation={group_counts[1]}, Test={group_counts[2]}")
    print(f"Image Allocation: Train={split_img_counts.get('train', 0)}, Val={split_img_counts.get('validation', 0)}, Test={split_img_counts.get('test', 0)}")
    print(f"Zero Leakage:   PASSED")
    print(f"\nArtifacts Saved:")
    print(f"  * Manifest:    {manifest_path}")
    print(f"  * Assignments: {assignment_path}")
    print(f"  * Summary:     {summary_path}")
    print(f"  * Report:      {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
