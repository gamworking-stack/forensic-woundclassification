#!/usr/bin/env python3
"""T23: validated metadata of the 23 Oct 2026 relabelled sheet, for a rerun of every experiment.

The sheet differs from the 30 Sep one the original study used in three ways, handled here:

1. ``Inclusion``: rows marked FALSE are excluded; so are photos whose label is outside the 7 classes
   (``imprint``, 7 photos without CVAT boxes).
2. Labels: a few photos changed class. Their raw image and CVAT boxes are still filed under the old
   label on Raptor, so each photo keeps the raw label it was filed under in ``source_label``
   (``find_source_image`` and ``manifest_key`` read it).
3. Cases: ``pseudonimized_id`` was renumbered and regrouped from case 316 on, and the sheet adds a
   ``patient_id`` that does not always agree with it. Photos sharing a case or a patient are joined
   into one group (connected components), and that group becomes the ``pseudonimized_id`` every
   later step splits, folds and bootstraps by; the sheet's own ID is kept as
   ``source_pseudonimized_id``. A placeholder ``patient_id`` (``n/a``) joins nothing. ``patient_id``
   and the demographic columns are never written.

Outputs in ``--output-dir``: ``validated_metadata.csv`` and ``metadata_validation_report.json``
(the T01 checks on the included photos), ``excluded_photos.csv`` and ``relabel_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.validate_metadata import LABEL_NORMALIZATION_MAP, validate_metadata  # noqa: E402


SOURCE_ID = "source_pseudonimized_id"
OUTPUT_COLUMNS = ["pseudonimized_id", "wound_id", "filename", "wound_label", "canonical_wound_label",
                  "relative_path", SOURCE_ID, "source_label"]
INCLUSION_VALUES = {"TRUE": True, "FALSE": False}
PATIENT_PLACEHOLDERS = {"", "n/a", "na", "-"}  # no patient ID recorded


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the T23 relabelled metadata (23 Oct 2026 sheet).")
    parser.add_argument("--input-csv", type=Path, default=Path("../thesis_data_23_oct.csv"),
                        help="The relabelled sheet exported as CSV.")
    parser.add_argument("--reference-metadata", type=Path, default=Path("data/validated_metadata.csv"),
                        help="The original validated metadata: where each photo is filed on Raptor.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/relabel_2026_10_23"))
    parser.add_argument("--exclude-labels", nargs="*", default=["imprint"],
                        help="Raw labels outside the 7 classes to exclude (decided 2026-10-23: imprint).")
    return parser.parse_args(argv)


def read_sheet(path: Path) -> Tuple[pd.DataFrame, int]:
    """The sheet's photo rows (all text, stripped) and the number of blank rows skipped.

    The Excel export runs to the sheet's last row; rows with no filename are padding.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda col: col.str.strip())
    required = {"pseudonimized_id", "patient_id", "filename", "wound_label", "Inclusion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    photos = df[df["filename"] != ""].reset_index(drop=True)
    # Every photo row must say TRUE or FALSE
    bad = sorted(set(photos["Inclusion"].str.upper()) - set(INCLUSION_VALUES))
    if bad:
        raise ValueError(f"Unexpected Inclusion values: {bad}")
    return photos, len(df) - len(photos)


def case_groups(df: pd.DataFrame) -> pd.Series:
    """Group ID of every row: photos sharing a ``pseudonimized_id`` or a ``patient_id`` are one group.

    The ID is the smallest source case ID in the group, so groups of one case keep their number.
    """
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(node: Tuple[str, str]) -> Tuple[str, str]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    # Join each row's case to its patient; a blank or placeholder patient_id joins nothing
    for case, patient in zip(df["pseudonimized_id"], df["patient_id"]):
        root = find(("case", case))
        if patient.lower() not in PATIENT_PLACEHOLDERS:
            parent[find(("patient", patient))] = root
    # Name each group by its smallest case ID
    smallest: Dict[Tuple[str, str], int] = {}
    for case in df["pseudonimized_id"]:
        root = find(("case", case))
        smallest[root] = min(smallest.get(root, int(case)), int(case))
    return df["pseudonimized_id"].map(lambda case: smallest[find(("case", case))])


def match_reference(df: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Raw label each photo is filed under in the original metadata (its folder and CVAT task).

    A photo matches the reference row with the same filename and raw label, or, if its label
    changed, the only reference row with that filename. Each reference row is used at most once.
    """
    by_name_label = reference.groupby(["filename", "wound_label"]).groups
    by_name = reference.groupby("filename").groups
    matched: List[int] = []
    problems: List[str] = []
    for row in df.to_dict("records"):
        # Same filename and label; a filename repeated under one label must also share the case
        candidates = list(by_name_label.get((row["filename"], row["wound_label"]), []))
        if len(candidates) > 1:
            candidates = [i for i in candidates if reference.at[i, "pseudonimized_id"] == row[SOURCE_ID]]
        # Otherwise the label changed: the filename must be unique in the reference
        if not candidates:
            candidates = list(by_name.get(row["filename"], []))
        if len(candidates) != 1:
            problems.append(f"{row[SOURCE_ID]}/{row['filename']} ({row['wound_label']}): {len(candidates)} matches")
            continue
        matched.append(candidates[0])
    if problems:
        raise ValueError(f"{len(problems)} photos do not match one reference row: {problems[:10]}")
    repeated = [i for i, n in Counter(matched).items() if n > 1]
    if repeated:
        raise ValueError(f"Reference rows matched twice: {reference.loc[repeated, 'filename'].tolist()}")
    return pd.Series(reference.loc[matched, "wound_label"].to_numpy(), index=df.index)


def prepare(sheet: pd.DataFrame, reference: pd.DataFrame,
            exclude_labels: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Included photos with groups and source labels, the excluded photos, and the counts."""
    sheet = sheet.copy()
    sheet[SOURCE_ID] = sheet["pseudonimized_id"]
    # Groups over every photo of the sheet, so an excluded photo can still join two cases
    sheet["pseudonimized_id"] = case_groups(sheet)

    # Exclusions: Inclusion FALSE first, then labels outside the 7 classes
    included = sheet["Inclusion"].str.upper().map(INCLUSION_VALUES)
    excluded_label = sheet["wound_label"].str.lower().isin([label.lower() for label in exclude_labels])
    reason = pd.Series("", index=sheet.index)
    reason[excluded_label] = "label_outside_7_classes"
    reason[~included] = "inclusion_false"
    excluded = sheet.loc[reason != "", [SOURCE_ID, "wound_id", "filename", "wound_label"]].assign(
        reason=reason[reason != ""])
    kept = sheet[reason == ""].reset_index(drop=True)

    # Where each kept photo is filed, and which photos changed class since annotation
    reference = reference.astype(str)
    kept["source_label"] = match_reference(kept, reference)
    canonical = lambda labels: labels.str.lower().map(LABEL_NORMALIZATION_MAP)  # noqa: E731
    changed = canonical(kept["source_label"]) != canonical(kept["wound_label"])
    relabelled = kept.loc[changed, [SOURCE_ID, "filename", "source_label", "wound_label"]].rename(
        columns={"source_label": "from", "wound_label": "to"})

    # Groups that join several source cases (by case renumbering or by patient)
    sources_per_group = kept.groupby("pseudonimized_id")[SOURCE_ID].nunique()
    report = {
        "sheet_photos": int(len(sheet)),
        "inclusion": {k.lower(): int(v) for k, v in sheet["Inclusion"].str.upper().value_counts().items()},
        "excluded_by_reason": {k: int(v) for k, v in excluded["reason"].value_counts().items()},
        "excluded_by_label": {k: int(v) for k, v in excluded["wound_label"].value_counts().items()},
        "included_photos": int(len(kept)),
        "included_photos_without_patient_id": int(kept["patient_id"].str.lower().isin(PATIENT_PLACEHOLDERS).sum()),
        "included_source_cases": int(kept[SOURCE_ID].nunique()),
        "included_groups": int(kept["pseudonimized_id"].nunique()),
        "groups_joining_several_source_cases": int((sources_per_group > 1).sum()),
        "photos_in_joined_groups": int(kept["pseudonimized_id"].isin(sources_per_group[sources_per_group > 1].index).sum()),
        "relabelled_photos": relabelled.to_dict("records"),
    }
    return kept, excluded.reset_index(drop=True), report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sheet, blank_rows = read_sheet(args.input_csv)
    reference = pd.read_csv(args.reference_metadata, dtype=str, keep_default_na=False)
    kept, excluded, report = prepare(sheet, reference, args.exclude_labels)

    # The T01 checks (7 classes, known labels, counts) on the included photos
    validated, validation = validate_metadata(kept)
    if validation["status"] != "passed":
        raise SystemExit(f"Validation failed: {validation['errors']}")
    # relative_path names the photo; the file itself is found through source_label
    if validated["relative_path"].duplicated().any():
        raise SystemExit(f"Duplicate relative_path: {validated.loc[validated['relative_path'].duplicated(), 'relative_path'].tolist()}")

    # Write the metadata (no patient_id or demographics), the exclusions and both reports
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validated[OUTPUT_COLUMNS].to_csv(args.output_dir / "validated_metadata.csv", index=False)
    excluded.to_csv(args.output_dir / "excluded_photos.csv", index=False)
    (args.output_dir / "metadata_validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {"input_csv": args.input_csv.name, "blank_rows_skipped": blank_rows,
              "reference_metadata": str(args.reference_metadata), **report}
    (args.output_dir / "relabel_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{report['included_photos']} photos included, {len(excluded)} excluded {report['excluded_by_reason']}")
    print(f"{report['included_source_cases']} source cases -> {report['included_groups']} groups "
          f"({report['groups_joining_several_source_cases']} join several cases)")
    print(f"{len(report['relabelled_photos'])} photos changed class since annotation")
    print(f"Per class: {validation['per_class_image_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
