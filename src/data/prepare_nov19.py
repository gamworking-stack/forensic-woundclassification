#!/usr/bin/env python3
"""T38: the 19 Nov 2026 dataset from the researcher's workbook.

The researcher keeps one row per photo in ``../Current dataset T23 (1,032 photos).xlsx`` (sheet
``Photos``): the T23 photos plus the photos new in the 19 Nov 2026 CVAT annotation, with the sheet
attributes, an anonymous patient number and the split chosen on 23 Nov 2026 (column
``New split (by patient)``). A row whose split is train, validation or test is in the dataset; the
others say why not (no 19 Nov boxes, Inclusion FALSE, duplicate photo).

The new photos' file names hold forensic case numbers, so the outputs go to a folder outside the
repository (``WOUND_DATA_DIR``), never into ``data/``. They follow the T23 layout, so every later
step (split, crops, CV folds, boxes, training) runs unchanged with ``WOUND_DATA_DIR`` pointing there:

- ``validated_metadata.csv``: the T23 columns. ``pseudonimized_id`` is the anonymous patient number,
  the unit of the split, the CV folds and the bootstrap. ``source_pseudonimized_id`` is the sheet case;
  a photo that is in no sheet gets ``10000 + patient`` (an integer no sheet case reaches).
- ``workbook_split.csv``: the workbook's split, which ``create_case_split.py`` must reproduce.
- ``splits/patient_cases.csv``: ``patient_case`` = the patient number (T25 layout).
- ``sheet.csv``: the attributes in the layout of the researcher's sheet, for the subgroup scripts
  (``patient_id`` holds the anonymous patient number, not the hospital number).
- ``image_sizes.csv``: EXIF-rotated photo sizes read from the local copies, the reference manifest
  of the dry-run crops (the Raptor crop job checks them against the photos there).
- ``raptor_upload.csv``: the photos that are not on Raptor yet, with their local file and raw folder.
- ``excluded_photos.csv`` and ``prepare_report.json``.

T39: ``--merge "cut wound" "penetration wound" --into "sharp injury"`` gives the 6-class version (photos,
patients and boxes unchanged; the annotation's own class stays in ``source_label``, so crops still find
their boxes). Its split is searched again, so ``workbook_split.csv`` is not written and
``--patient-cases-from-split`` writes ``splits/patient_cases.csv`` after ``create_case_split.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.data.prepare_relabel import OUTPUT_COLUMNS, SOURCE_ID  # noqa: E402
from src.data.validate_metadata import LABEL_NORMALIZATION_MAP  # noqa: E402

SPLITS = ("train", "validation", "test")
NEW_SOURCE_OFFSET = 10_000  # source case of a photo in no sheet: offset + patient number
SHEET_COLUMNS = ["pseudonimized_id", "patient_id", "wound_id", "filename", "skin_tone", "wound_label", "tattoo",
                 "body_site", "autopsy", "race_ethnicity", "age", "Inclusion", "number_classes"]
# Workbook column -> sheet column (the "(sheet)" columns hold the raw sheet values)
SHEET_SOURCES = {"skin_tone": "Skin tone, MST (sheet)", "tattoo": "Tattoo (sheet)", "body_site": "Body site (sheet)",
                 "autopsy": "Autopsy (sheet)", "race_ethnicity": "Race/ethnicity (sheet)", "age": "Age (sheet)",
                 "Inclusion": "Inclusion (sheet)", "number_classes": "Number of classes (sheet)"}
BODY_SITE_SPELLING = {"multiple": "multi"}  # the workbook's spelling -> the sheet's
EXIF_ORIENTATION = 0x0112
# The local copy of Raptor's raw tree (raw_data/wound_ml_7classes): photos filed there are on Raptor already
RAW_TREE_COPY = unicodedata.normalize("NFC", "/ฝากรูป/wound ml/")
REQUIRED = ["No.", "In current dataset", "Class", "File name", "Patient (anonymous no.)", "Case in sheet", "Wound ID",
            "Label in sheet", "Folder / CVAT label", "Image file (from Desktop)", "New split (by patient)",
            *SHEET_SOURCES.values()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the T38 dataset (19 Nov 2026 annotation) from the workbook.")
    parser.add_argument("--workbook", type=Path, default=Path("../Current dataset T23 (1,032 photos).xlsx"))
    parser.add_argument("--sheet-name", default="Photos")
    parser.add_argument("--photo-root", type=Path, default=Path.home() / "Desktop",
                        help="Folder the workbook's 'Image file (from Desktop)' paths start from.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--merge", nargs="+", default=[], metavar="CLASS",
                        help="Canonical classes merged into --into (T39: 'cut wound' 'penetration wound').")
    parser.add_argument("--into", default=None, help="Name of the merged class (T39: 'sharp injury').")
    parser.add_argument("--patient-cases-from-split", action="store_true",
                        help="Only write splits/patient_cases.csv from splits/case_level_split_manifest.csv.")
    args = parser.parse_args(argv)
    if bool(args.merge) != bool(args.into):
        parser.error("--merge and --into go together")
    return args


def text(value: Any) -> str:
    """A cell as text: blank for empty, integers without '.0'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def read_workbook(path: Path, sheet_name: str) -> pd.DataFrame:
    """The workbook's photo rows (rows without a 'No.' are padding)."""
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} sheet {sheet_name!r} is missing columns: {missing}")
    return df[df["No."].notna()].reset_index(drop=True)


def canonical(label: str) -> Optional[str]:
    return LABEL_NORMALIZATION_MAP.get(label.strip().lower())


def image_size(path: Path) -> tuple:
    """Width and height after EXIF rotation, without decoding the pixels."""
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(EXIF_ORIENTATION) in (5, 6, 7, 8):  # rotated by 90 or 270 degrees
            width, height = height, width
    return width, height


def prepare(workbook: pd.DataFrame, photo_root: Optional[Path],
            merge: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """All outputs as data frames plus the report; ``photo_root=None`` skips the image sizes.

    ``merge`` maps canonical classes to a merged class (T39); labels and checks follow it.
    """
    merge = merge or {}
    merged = lambda label: merge.get(label, label)  # noqa: E731
    split = workbook["New split (by patient)"].map(text)
    kept = workbook[split.isin(SPLITS)].copy()
    excluded = workbook[~split.isin(SPLITS)]

    rows: List[Dict[str, str]] = []
    for _, r in kept.iterrows():
        patient = text(r["Patient (anonymous no.)"])
        if not patient.isdigit():
            raise ValueError(f"No. {text(r['No.'])} ({r['File name']}): patient number {patient!r} is not a number")
        folder_label = text(r["Folder / CVAT label"])
        sheet_label = text(r["Label in sheet"]) or folder_label  # photos in no sheet: the annotation's class
        case = text(r["Case in sheet"]) or str(NEW_SOURCE_OFFSET + int(patient))
        cls = merged(text(r["Class"]))
        if canonical(folder_label) is None or merged(canonical(sheet_label) or "") != cls:
            raise ValueError(f"No. {text(r['No.'])} ({r['File name']}): labels {sheet_label!r}/{folder_label!r} "
                             f"do not give class {cls!r}")
        rows.append({"pseudonimized_id": patient, "wound_id": text(r["Wound ID"]), "filename": text(r["File name"]),
                     "wound_label": sheet_label, "canonical_wound_label": cls,
                     "relative_path": f"{cls}/{text(r['File name'])}", SOURCE_ID: case, "source_label": folder_label,
                     "split": text(r["New split (by patient)"]), "in_current": text(r["In current dataset"]),
                     "local_file": text(r["Image file (from Desktop)"])})
    photos = pd.DataFrame(rows)

    # One photo per (annotation class, file stem), the key the crops join on; one patient per split
    key = photos["source_label"].map(canonical) + "/" + photos["filename"].map(lambda f: Path(f).stem)
    if key.duplicated().any():
        raise ValueError(f"Photos sharing a class and file name: {sorted(key[key.duplicated()])}")
    if photos.duplicated(["relative_path"]).any():
        raise ValueError("Two photos share a relative_path")
    spread = photos.groupby("pseudonimized_id")["split"].nunique()
    if (spread > 1).any():
        raise ValueError(f"Patients in more than one split: {sorted(spread[spread > 1].index)}")
    if photos.duplicated([SOURCE_ID, "filename"]).any():
        raise ValueError("Two photos share a (sheet case, filename)")

    metadata = photos[OUTPUT_COLUMNS]
    patient_cases = photos[["pseudonimized_id", SOURCE_ID, "filename", "canonical_wound_label", "relative_path",
                            "split"]].assign(patient_case=photos["pseudonimized_id"].astype(int))

    sheet = pd.DataFrame({"pseudonimized_id": photos[SOURCE_ID], "patient_id": photos["pseudonimized_id"],
                          "wound_id": photos["wound_id"], "filename": photos["filename"],
                          "wound_label": photos["wound_label"]})
    for column, source in SHEET_SOURCES.items():
        sheet[column] = kept[source].map(text).to_numpy()
    sheet["body_site"] = sheet["body_site"].map(lambda s: BODY_SITE_SPELLING.get(s.lower(), s))
    sheet = sheet[SHEET_COLUMNS]
    not_true = sheet["Inclusion"].str.upper() != "TRUE"
    if not_true.any():
        raise ValueError(f"Photos in the split without Inclusion TRUE: {sorted(sheet.loc[not_true, 'filename'])}")

    sizes = pd.DataFrame()
    if photo_root is not None:
        missing = [f for f in photos["local_file"] if not f or not (photo_root / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} photos not found under {photo_root}: {missing[:5]}")
        dims = [image_size(photo_root / f) for f in photos["local_file"]]
        sizes = photos[["filename", "source_label", "canonical_wound_label"]].assign(
            source_width=[w for w, _ in dims], source_height=[h for _, h in dims])

    # Photos new to Raptor: those that are not in the T23 raw tree (only the T23 photos and the sheet's
    # excluded photos are there). The raw tree is filed by the sheet's raw label
    new = photos[photos["in_current"] != "Yes"]
    on_raptor = new["local_file"].map(lambda f: RAW_TREE_COPY in unicodedata.normalize("NFC", f))
    upload = new[~on_raptor][["filename", "source_label", "local_file"]]

    report = {
        "workbook_rows": len(workbook),
        "photos": len(photos),
        "photos_from_t23": int((photos["in_current"] == "Yes").sum()),
        "photos_new": int((photos["in_current"] != "Yes").sum()),
        "photos_in_no_sheet": int((photos[SOURCE_ID].astype(int) >= NEW_SOURCE_OFFSET).sum()),
        "patients": int(photos["pseudonimized_id"].nunique()),
        "patients_by_split": {s: int(photos.loc[photos["split"] == s, "pseudonimized_id"].nunique()) for s in SPLITS},
        "photos_by_split": {s: int((photos["split"] == s).sum()) for s in SPLITS},
        "photos_by_class": dict(sorted(Counter(photos["canonical_wound_label"]).items())),
        "excluded": dict(sorted(Counter(excluded["New split (by patient)"].map(text)).items())),
        "relabelled_photos": int((photos["wound_label"].map(canonical) != photos["source_label"].map(canonical)).sum()),
        "merged_classes": {k: v for k, v in merge.items()},
        "photos_to_upload": len(upload),
    }
    excluded_out = pd.DataFrame({"filename": excluded["File name"].map(text), "class": excluded["Class"].map(text),
                                 "in_current": excluded["In current dataset"].map(text),
                                 "reason": excluded["New split (by patient)"].map(text)})
    return {"metadata": metadata, "workbook_split": photos[["pseudonimized_id", "relative_path", "split"]],
            "patient_cases": patient_cases, "sheet": sheet, "image_sizes": sizes, "raptor_upload": upload,
            "excluded": excluded_out, "report": report}


def patient_cases_from_split(out: Path) -> pd.DataFrame:
    """T25 layout from the final split manifest: the patient number is the group."""
    manifest = pd.read_csv(out / "splits/case_level_split_manifest.csv", dtype=str, keep_default_na=False)
    return manifest[["pseudonimized_id", SOURCE_ID, "filename", "canonical_wound_label", "relative_path",
                     "split"]].assign(patient_case=manifest["pseudonimized_id"].astype(int))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    if out.resolve() == Path("data").resolve() or Path("data").resolve() in out.resolve().parents:
        print(f"Refusing to write {out}: the T38 lists hold case numbers and stay outside the repository "
              "(source scripts/hpc/nov19_env.sh first).", file=sys.stderr)
        return 1
    if args.patient_cases_from_split:
        patient_cases_from_split(out).to_csv(out / "splits/patient_cases.csv", index=False)
        print(f"Wrote {out / 'splits/patient_cases.csv'}")
        return 0
    merge = {c: args.into for c in args.merge}
    result = prepare(read_workbook(args.workbook, args.sheet_name), args.photo_root, merge)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    result["metadata"].to_csv(out / "validated_metadata.csv", index=False)
    if not merge:  # the workbook's split is the 7-class one; a merged version is split again
        result["workbook_split"].to_csv(out / "workbook_split.csv", index=False)
        result["patient_cases"].to_csv(out / "splits/patient_cases.csv", index=False)
    result["sheet"].to_csv(out / "sheet.csv", index=False)
    result["image_sizes"].to_csv(out / "image_sizes.csv", index=False)
    result["raptor_upload"].to_csv(out / "raptor_upload.csv", index=False)
    result["excluded"].to_csv(out / "excluded_photos.csv", index=False)
    report = {"workbook": str(args.workbook.resolve()), **result["report"]}
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
