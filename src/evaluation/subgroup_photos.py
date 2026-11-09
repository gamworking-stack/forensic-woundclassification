#!/usr/bin/env python3
"""The photos of one sheet subgroup (e.g. tattoo) in every split, as a list for gradcam_panels.py (T31).

Runs on the laptop, where the researcher's sheet stays. The output holds only the photos' manifest
paths, split and wound class (no sheet attribute), so it can be copied to Raptor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.data.prepare_relabel import read_sheet  # noqa: E402
from src.evaluation.subgroup import ATTRIBUTES, subgroup_columns  # noqa: E402

SPLITS = ("train", "validation", "test")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photos of one sheet subgroup, every split.")
    parser.add_argument("--sheet", type=Path, required=True, help="The researcher's sheet as CSV (as subgroup.py).")
    parser.add_argument("--attribute", choices=ATTRIBUTES, default="tattoo")
    parser.add_argument("--level", default="tattoo", help="Subgroup label, as subgroup.py names it.")
    parser.add_argument("--split-manifest", type=Path, default=DATA_DIR / "splits/case_level_split_manifest.csv")
    parser.add_argument("--output", type=Path, required=True, help="CSV, outside the repository.")
    return parser.parse_args(argv)


def subgroup_photos(sheet_path: Path, split_manifest: Path, attribute: str, level: str) -> pd.DataFrame:
    """Manifest photos of every split whose sheet row falls in ``level`` of ``attribute``."""
    sheet, _ = read_sheet(sheet_path)
    sheet = sheet[sheet["Inclusion"].str.upper() == "TRUE"]
    photos = pd.read_csv(split_manifest, dtype=str, keep_default_na=False)
    # The sheet's own ID sits beside the manifest's group ID (T23), as in subgroup.test_photos
    source = "source_pseudonimized_id" if "source_pseudonimized_id" in photos else "pseudonimized_id"
    labels = pd.concat([sheet[["pseudonimized_id", "filename"]], subgroup_columns(sheet)[[attribute]]], axis=1)
    merged = photos.merge(labels.rename(columns={"pseudonimized_id": source}), on=[source, "filename"],
                          how="left", validate="one_to_one")
    if merged[attribute].isna().any():
        raise ValueError(f"{int(merged[attribute].isna().sum())} manifest photos are not in the sheet as included")
    chosen = merged[merged[attribute] == level].copy()
    chosen["split"] = pd.Categorical(chosen["split"], SPLITS, ordered=True)
    return chosen.sort_values(["split", "canonical_wound_label", "relative_path"])[
        ["relative_path", "split", "canonical_wound_label"]]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    photos = subgroup_photos(args.sheet, args.split_manifest, args.attribute, args.level)
    if photos.empty:
        raise ValueError(f"No photo has {args.attribute} = {args.level!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    photos.to_csv(args.output, index=False)
    print(f"{len(photos)} photos with {args.attribute} = {args.level}: "
          + ", ".join(f"{s} {n}" for s, n in photos["split"].value_counts(sort=False).items()) + f" -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
