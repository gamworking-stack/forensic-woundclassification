"""T25: one case per patient, and prediction files rewritten with it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.data.patient_cases import PATIENT_CASE, patient_cases
from src.evaluation.metrics import load_predictions
from src.evaluation.patient_rescore import main as rescore_main


# (group, sheet case, file name, class, patient_id, split): group 10 joins sheet cases 10 and 11 and holds
# patients X and Y; group 20 has no patient ID; group 30 is a train photo
PHOTOS = [("10", "10", "A", "contusion", "X", "test"), ("10", "10", "B", "contusion", "Y", "test"),
          ("10", "11", "C", "cut wound", "Y", "test"), ("20", "20", "D", "cut wound", "n/a", "test"),
          ("30", "30", "E", "contusion", "Z", "train")]
RAW_FOLDER = {"contusion": "contusion", "cut wound": "cut"}


def sheet_and_manifest():
    sheet = pd.DataFrame([{"pseudonimized_id": case, "patient_id": pid, "filename": f"{name}.JPG",
                           "wound_label": RAW_FOLDER[label], "Inclusion": "TRUE"}
                          for _, case, name, label, pid, _ in PHOTOS])
    # An excluded row of another patient must not count
    sheet = pd.concat([sheet, pd.DataFrame([{**sheet.iloc[0], "patient_id": "W", "Inclusion": "FALSE",
                                             "wound_label": "cut"}])], ignore_index=True)
    manifest = pd.DataFrame([{"pseudonimized_id": group, "filename": f"{name}.JPG", "canonical_wound_label": label,
                              "relative_path": f"{label}/{name}.JPG", "source_pseudonimized_id": case, "split": split}
                             for group, case, name, label, _, split in PHOTOS])
    return sheet, manifest


def test_patients_are_numbered_within_groups():
    sheet, manifest = sheet_and_manifest()
    table = patient_cases(sheet, manifest).set_index("filename")
    # X first (group 10, case 10, photo A), then Y; D's sheet case stands for its patient
    assert table.loc["A.JPG", PATIENT_CASE] == 1
    assert table.loc["B.JPG", PATIENT_CASE] == table.loc["C.JPG", PATIENT_CASE] == 2
    assert table.loc["D.JPG", PATIENT_CASE] == 3 and table.loc["E.JPG", PATIENT_CASE] == 4
    assert "patient_id" not in table.columns


def test_patient_in_two_groups_is_refused():
    sheet, manifest = sheet_and_manifest()
    sheet.loc[sheet["filename"] == "E.JPG", "patient_id"] = "X"
    with pytest.raises(ValueError, match="more than one T23 group"):
        patient_cases(sheet, manifest)


def prediction_file(path: Path, rows) -> None:
    # (relative_path, group, class) rows of a test prediction file
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"relative_path": rel, "pseudonimized_id": group, "split": "test", "true_class": label,
                   "predicted_class": label, "prob_contusion": float(label == "contusion"),
                   "prob_cut_wound": float(label == "cut wound")} for rel, group, label in rows]
                 ).to_csv(path, index=False)


@pytest.fixture()
def runs(tmp_path):
    sheet, manifest = sheet_and_manifest()
    patient_cases(sheet, manifest).to_csv(tmp_path / "patient_cases.csv", index=False)
    manifest.to_csv(tmp_path / "split.csv", index=False)
    test = [p for p in PHOTOS if p[5] == "test"]
    pd.DataFrame([{"relative_path": f"{label}/{name}.JPG", "roi_relative_path": f"test/{label}/{name}_roi.jpg"}
                  for _, _, name, label, _, _ in test]).to_csv(tmp_path / "roi.csv", index=False)
    pd.DataFrame([{"relative_path": f"{label}/{name}.JPG", "roi_relative_path": f"test/{label}/{name}_bbox01.jpg"}
                  for _, _, name, label, _, _ in test]).to_csv(tmp_path / "box.csv", index=False)
    src = tmp_path / "src"
    # Full images name the raw photo under its raw folder; ROI and crops name the crop
    prediction_file(src / "main/full/m/test_predictions.csv",
                    [(f"{RAW_FOLDER[label]}/{name}.JPG", group, label) for group, _, name, label, _, _ in test])
    prediction_file(src / "main/roi/m/test_predictions.csv",
                    [(f"test/{label}/{name}_roi.jpg", group, label) for group, _, name, label, _, _ in test])
    prediction_file(src / "roi_crop/crop/m/test_predictions.csv",
                    [(f"test/{label}/{name}_bbox01.jpg", group, label) for group, _, name, label, _, _ in test])
    args = ["--source-root", str(src), "--output-root", str(tmp_path / "out"),
            "--patient-cases", str(tmp_path / "patient_cases.csv"), "--split-manifest", str(tmp_path / "split.csv"),
            "--roi-manifest", str(tmp_path / "roi.csv"), "--box-manifest", str(tmp_path / "box.csv")]
    return tmp_path, args


def test_prediction_files_get_patient_cases(runs):
    tmp_path, args = runs
    assert rescore_main(args) == 0
    for rel in ("main/full/m", "main/roi/m", "roi_crop/crop/m"):
        p = load_predictions(tmp_path / "out" / rel / "test_predictions.csv", "test")
        # Group 10 becomes patients 1 and 2, so the bootstrap now draws 3 cases instead of 2
        assert list(p.cases) == ["1", "2", "2", "3"]
    manifest = pd.read_csv(tmp_path / "out/split_manifest.csv", dtype=str).set_index("filename")
    assert manifest.loc["C.JPG", "pseudonimized_id"] == "2" and manifest.loc["E.JPG", "pseudonimized_id"] == "4"


def test_row_of_another_group_is_refused(runs):
    tmp_path, args = runs
    path = tmp_path / "src/main/roi/m/test_predictions.csv"
    frame = pd.read_csv(path, dtype=str)
    frame.loc[0, "pseudonimized_id"] = "20"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="another group"):
        rescore_main(args)


def test_output_inside_repository_is_refused(runs):
    _, args = runs
    args[args.index("--output-root") + 1] = str(Path(__file__).resolve().parents[1] / "results" / "patient")
    with pytest.raises(ValueError, match="inside the repository"):
        rescore_main(args)
