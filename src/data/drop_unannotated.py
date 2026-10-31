#!/usr/bin/env python3
"""T26: drop the photos without a CVAT box from every manifest of a dataset version.

The crop step lists the included photos that no export annotates (``roi/crop_summary.json``,
``unannotated_records``). No model ever saw them: training and testing keep only full images with a
crop (``full_scope = matched``). This removes them from the metadata, the split manifest and the
patient cases, adds them to ``excluded_photos.csv`` (reason ``no_annotation``), drops groups left
empty from the case assignments and CV folds, and recounts the split summary and reports. The split
is not searched again: every other photo keeps its split and fold, so every run stays reproducible.

Running it again changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_case_split import generate_summary  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402
from src.data.prepare_relabel import SOURCE_ID  # noqa: E402


REASON = "no_annotation"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop photos without a CVAT box from a dataset version.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Dataset version to edit in place (e.g. data/relabel_2026_10_23).")
    return parser.parse_args(argv)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def drop_unannotated(data_dir: Path) -> pd.DataFrame:
    """Edit ``data_dir`` in place; return the photos dropped by this call (none on a rerun)."""
    records = json.loads((data_dir / "roi/crop_summary.json").read_text(encoding="utf-8"))["unannotated_records"]
    unannotated = {(r["pseudonimized_id"], r["filename"]) for r in records}

    def keep(frame: pd.DataFrame) -> pd.Series:
        return ~pd.Series(list(zip(frame["pseudonimized_id"], frame["filename"])), index=frame.index).isin(unannotated)

    metadata = read(data_dir / "validated_metadata.csv")
    dropped = metadata[~keep(metadata)]
    metadata[keep(metadata)].to_csv(data_dir / "validated_metadata.csv", index=False)
    splits = data_dir / "splits"
    manifest = read(splits / "case_level_split_manifest.csv")
    manifest = manifest[keep(manifest)]
    manifest.to_csv(splits / "case_level_split_manifest.csv", index=False)
    if (splits / "patient_cases.csv").is_file():
        patients = read(splits / "patient_cases.csv")
        patients[keep(patients)].to_csv(splits / "patient_cases.csv", index=False)

    # Groups left without a photo leave the case assignments and the CV folds
    groups = set(manifest["pseudonimized_id"])
    for name in ("case_split_assignments.csv", "cv_folds.csv"):
        table = read(splits / name)
        table[table["pseudonimized_id"].isin(groups)].to_csv(splits / name, index=False)

    excluded = read(data_dir / "excluded_photos.csv")
    new = dropped.assign(reason=REASON)[excluded.columns.drop("reason").tolist() + ["reason"]]
    excluded = pd.concat([excluded, new], ignore_index=True)
    excluded.to_csv(data_dir / "excluded_photos.csv", index=False)

    # Recount the summary and reports; the split's search settings and score stay as built
    generate_summary(manifest, "pseudonimized_id", "canonical_wound_label").to_csv(
        splits / "case_level_split_summary.csv", index=False)
    report = json.loads((splits / "case_level_split_report.json").read_text(encoding="utf-8"))
    report.update({"total_records": len(manifest), "total_unique_cases": len(groups),
                   "case_counts_by_split": manifest.groupby("split")["pseudonimized_id"].nunique().to_dict(),
                   "image_counts_by_split": manifest["split"].value_counts().to_dict(),
                   "dropped_no_annotation": int((excluded["reason"] == REASON).sum())})
    write_json(splits / "case_level_split_report.json", report)
    folds = json.loads((splits / "cv_folds_report.json").read_text(encoding="utf-8"))
    folds["development_cases"] = len(read(splits / "cv_folds.csv"))
    write_json(splits / "cv_folds_report.json", folds)
    relabel = json.loads((data_dir / "relabel_report.json").read_text(encoding="utf-8"))
    relabel["excluded_by_reason"][REASON] = int((excluded["reason"] == REASON).sum())
    relabel["excluded_by_label"] = excluded["wound_label"].value_counts().to_dict()
    joined = manifest.groupby("pseudonimized_id")[SOURCE_ID].transform("nunique") > 1
    relabel.update({"included_photos": len(metadata[keep(metadata)]),
                    "included_source_cases": manifest[SOURCE_ID].nunique(), "included_groups": len(groups),
                    "groups_joining_several_source_cases": manifest.loc[joined, "pseudonimized_id"].nunique(),
                    "photos_in_joined_groups": int(joined.sum())})
    write_json(data_dir / "relabel_report.json", relabel)
    return dropped


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    dropped = drop_unannotated(args.data_dir)
    manifest = read(args.data_dir / "splits/case_level_split_manifest.csv")
    print(f"Dropped {len(dropped)} photos without a box; {len(manifest)} photos in "
          f"{manifest['pseudonimized_id'].nunique()} groups remain {manifest['split'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
