"""T38: the 19 Nov 2026 dataset prepared from the researcher's workbook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from PIL import Image

from src.data.prepare_nov19 import NEW_SOURCE_OFFSET, REQUIRED, image_size, main, prepare


def workbook(rows):
    # Rows: (No., in current, class, file, patient, case, sheet label, folder label, local file, split, body site)
    records = []
    for no, cur, cls, fname, patient, case, sheet_label, folder, local, split, site in rows:
        record = {c: None for c in REQUIRED}
        record.update({"No.": no, "In current dataset": cur, "Class": cls, "File name": fname,
                       "Patient (anonymous no.)": patient, "Case in sheet": case, "Wound ID": 1,
                       "Label in sheet": sheet_label, "Folder / CVAT label": folder,
                       "Image file (from Desktop)": local, "New split (by patient)": split,
                       "Inclusion (sheet)": "TRUE", "Age (sheet)": 30, "Race/ethnicity (sheet)": "Asian",
                       "Skin tone, MST (sheet)": 4, "Autopsy (sheet)": 1, "Body site (sheet)": site,
                       "Tattoo (sheet)": 0, "Number of classes (sheet)": 1})
        records.append(record)
    return pd.DataFrame(records)


RAW = "[Thesis] Wound classification/ฝากรูป/wound ml"
BASE = [
    (1, "Yes", "gunshot wound", "A1.JPG", 1, 5, "gsw", "gsw", f"{RAW}/gsw/A1.JPG", "train", "arm"),
    (2, "Yes", "penetration wound", "A2.JPG", 1, 5, "penetrating wound (sharp&blunt)", "gsw", f"{RAW}/gsw/A2.JPG",
     "train", "leg"),  # relabelled after annotation
    (3, "No", "scratch", "F20-0001 (3)", 2, None, None, "scratch", "รูปตามประเภท/scratch/F20-0001 (3).JPG", "test",
     "multiple"),  # new, in no sheet
    (4, "No", "graze wound", "B4.JPG", 3, 136, "graze", "graze", f"{RAW}/graze/B4.JPG", "validation", "face"),
    (5, "Yes", "cut wound", "C5.JPG", 4, 7, "cut", "cut", f"{RAW}/cut/C5.JPG", "excluded: no 19 Nov boxes", "arm"),
]


def test_prepare_builds_the_t23_layout_with_patients_as_groups():
    result = prepare(workbook(BASE), photo_root=None)
    meta = result["metadata"].set_index("filename")
    # Excluded rows drop out; the group is the patient; a photo in no sheet gets offset + patient as its case
    assert list(meta.index) == ["A1.JPG", "A2.JPG", "F20-0001 (3)", "B4.JPG"]
    assert meta.loc["A2.JPG", "pseudonimized_id"] == "1"
    assert meta.loc["F20-0001 (3)", "source_pseudonimized_id"] == str(NEW_SOURCE_OFFSET + 2)
    assert meta.loc["F20-0001 (3)", "wound_label"] == "scratch"
    assert meta.loc["F20-0001 (3)", "relative_path"] == "scratch/F20-0001 (3)"
    assert meta.loc["A2.JPG", "source_label"] == "gsw"
    assert result["report"]["relabelled_photos"] == 1
    assert result["excluded"]["filename"].tolist() == ["C5.JPG"]
    # patient_case is the patient number; the sheet export carries it as patient_id, never a hospital number
    assert result["patient_cases"]["patient_case"].tolist() == [1, 1, 2, 3]
    sheet = result["sheet"].set_index("filename")
    assert sheet.loc["F20-0001 (3)", "patient_id"] == "2"
    assert sheet.loc["F20-0001 (3)", "body_site"] == "multi"  # the subgroup scripts' spelling
    # Only the new photo outside the local copy of the Raptor raw tree is uploaded
    assert result["raptor_upload"]["filename"].tolist() == ["F20-0001 (3)"]


def test_prepare_refuses_a_patient_in_two_splits():
    rows = [list(r) for r in BASE]
    rows[1][9] = "test"
    with pytest.raises(ValueError, match="more than one split"):
        prepare(workbook([tuple(r) for r in rows]), photo_root=None)


def test_prepare_refuses_labels_that_do_not_give_the_class():
    rows = [list(r) for r in BASE]
    rows[0][2] = "laceration"
    with pytest.raises(ValueError, match="do not give class"):
        prepare(workbook([tuple(r) for r in rows]), photo_root=None)


def test_prepare_refuses_two_photos_with_one_annotation_key():
    rows = [list(r) for r in BASE] + [[6, "No", "gunshot wound", "A1", 5, None, None, "gsw", "x/A1.JPG", "train", "arm"]]
    with pytest.raises(ValueError, match="sharing a class and file name"):
        prepare(workbook([tuple(r) for r in rows]), photo_root=None)


def test_image_size_follows_the_exif_rotation(tmp_path):
    path = tmp_path / "p.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees
    Image.new("RGB", (40, 30)).save(path, exif=exif)
    assert image_size(path) == (30, 40)


def test_main_never_writes_into_the_repository_data_folder():
    assert main(["--output-dir", "data/relabel_2026_11_19"]) == 1
