#!/usr/bin/env python3
"""T25: one case per patient, for scoring the T23 runs with patients as the bootstrap unit.

T23 joins photos that share a sheet case or a ``patient_id`` into one group; 17 of the 336 groups hold
several patients because the sheet's case numbers are misaligned from case 316 on. The split and the
trained models stay as they are. This script only numbers the patients: every included photo gets a
``patient_case``, the same number for all photos of one ``patient_id``. A photo with no patient ID
(``n/a`` or blank) counts its sheet case as the patient. Every patient must sit inside one T23 group,
hence inside one split and one CV fold.

``patient_case`` runs from 1 in the order of each patient's smallest group, sheet case and photo, so it
is a pseudonym like ``pseudonimized_id``; ``patient_id`` itself is never written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.data.prepare_relabel import PATIENT_PLACEHOLDERS, SOURCE_ID, read_sheet  # noqa: E402


PATIENT_CASE = "patient_case"
OUTPUT_COLUMNS = ["pseudonimized_id", SOURCE_ID, "filename", "canonical_wound_label", "relative_path", "split",
                  PATIENT_CASE]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Number the patients of the T23 split (one case per patient).")
    parser.add_argument("--sheet", type=Path, required=True,
                        help="The researcher's sheet as CSV with patient_id (e.g. ../thesis_data_27_oct_corrected.csv).")
    parser.add_argument("--split-manifest", type=Path, default=DATA_DIR / "splits/case_level_split_manifest.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "splits/patient_cases.csv")
    return parser.parse_args(argv)


def patient_cases(sheet: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """The manifest's photos with their ``patient_case``; refuses a patient that crosses groups."""
    sheet = sheet[sheet["Inclusion"].str.upper() == "TRUE"]
    keys = sheet[["pseudonimized_id", "filename", "patient_id"]].rename(columns={"pseudonimized_id": SOURCE_ID})
    photos = manifest.merge(keys, on=[SOURCE_ID, "filename"], how="left", validate="one_to_one")
    if photos["patient_id"].isna().any():
        missing = photos.loc[photos["patient_id"].isna(), "relative_path"].head(3).tolist()
        raise ValueError(f"{int(photos['patient_id'].isna().sum())} photos are not in the sheet as included "
                         f"(e.g. {missing})")
    # No patient ID: the sheet case stands for the patient
    known = ~photos["patient_id"].str.strip().str.lower().isin(PATIENT_PLACEHOLDERS)
    patient = photos["patient_id"].where(known, "case:" + photos[SOURCE_ID])
    groups = patient.groupby(patient).transform(lambda s: photos.loc[s.index, "pseudonimized_id"].nunique())
    if (groups > 1).any():
        raise ValueError("A patient's photos are in more than one T23 group; the split would not keep them apart")
    # Number patients by their first group, sheet case and photo
    order = (photos.assign(_patient=patient, _group=photos["pseudonimized_id"].astype(int),
                           _case=photos[SOURCE_ID].astype(int))
             .groupby("_patient")[["_group", "_case", "relative_path"]].min()
             .sort_values(["_group", "_case", "relative_path"]))
    number = {p: i + 1 for i, p in enumerate(order.index)}
    photos[PATIENT_CASE] = patient.map(number)
    return photos[OUTPUT_COLUMNS]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sheet, _ = read_sheet(args.sheet)
    manifest = pd.read_csv(args.split_manifest, dtype=str, keep_default_na=False)
    table = patient_cases(sheet, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    per_split = table.groupby("split")[PATIENT_CASE].nunique().to_dict()
    print(f"{len(table)} photos, {table[PATIENT_CASE].nunique()} patients from "
          f"{table['pseudonimized_id'].nunique()} groups {per_split}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
