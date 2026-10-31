"""T26: photos without a CVAT box leave every manifest; the split of the others is unchanged."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.drop_unannotated import drop_unannotated


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_unannotated_photos_are_dropped_once(tmp_path):
    # Group 1 keeps photo A after B is dropped; group 2 (only C) disappears
    photos = [("1", "A.JPG", "train"), ("1", "B.JPG", "train"), ("2", "C.JPG", "validation"), ("3", "D.JPG", "test")]
    rows = [{"pseudonimized_id": g, "wound_id": "1", "filename": f, "wound_label": "cut", "canonical_wound_label": "cut wound",
             "relative_path": f"cut wound/{f}", "source_pseudonimized_id": g, "source_label": "cut"} for g, f, _ in photos]
    write(tmp_path / "validated_metadata.csv", rows)
    write(tmp_path / "splits/case_level_split_manifest.csv", [{**r, "split": s} for r, (_, _, s) in zip(rows, photos)])
    write(tmp_path / "splits/patient_cases.csv", [{**r, "split": s, "patient_case": str(i + 1)}
                                                  for i, (r, (_, _, s)) in enumerate(zip(rows, photos))])
    write(tmp_path / "splits/case_split_assignments.csv", [{"pseudonimized_id": g, "split": s} for g, s in
                                                           (("1", "train"), ("2", "validation"), ("3", "test"))])
    write(tmp_path / "splits/cv_folds.csv", [{"pseudonimized_id": "1", "fold": "1"}, {"pseudonimized_id": "2", "fold": "2"}])
    write(tmp_path / "excluded_photos.csv", [{"source_pseudonimized_id": "9", "wound_id": "1", "filename": "Z.JPG",
                                              "wound_label": "imprint", "reason": "label_outside_7_classes"}])
    (tmp_path / "roi").mkdir()
    (tmp_path / "roi/crop_summary.json").write_text(json.dumps({"unannotated_records": [
        {"pseudonimized_id": "1", "filename": "B.JPG"}, {"pseudonimized_id": "2", "filename": "C.JPG"}]}))
    for name in ("splits/case_level_split_report.json", "splits/cv_folds_report.json", "relabel_report.json"):
        (tmp_path / name).write_text(json.dumps({"excluded_by_reason": {}}))

    assert len(drop_unannotated(tmp_path)) == 2
    assert len(drop_unannotated(tmp_path)) == 0
    manifest = pd.read_csv(tmp_path / "splits/case_level_split_manifest.csv", dtype=str)
    assert list(manifest["filename"]) == ["A.JPG", "D.JPG"] and list(manifest["split"]) == ["train", "test"]
    assert list(pd.read_csv(tmp_path / "splits/patient_cases.csv", dtype=str)["patient_case"]) == ["1", "4"]
    assert list(pd.read_csv(tmp_path / "splits/cv_folds.csv", dtype=str)["pseudonimized_id"]) == ["1"]
    assert list(pd.read_csv(tmp_path / "splits/case_split_assignments.csv", dtype=str)["pseudonimized_id"]) == ["1", "3"]
    excluded = pd.read_csv(tmp_path / "excluded_photos.csv", dtype=str)
    assert excluded["reason"].value_counts().to_dict() == {"no_annotation": 2, "label_outside_7_classes": 1}
    report = json.loads((tmp_path / "relabel_report.json").read_text())
    assert report["included_photos"] == 2 and report["excluded_by_reason"] == {"no_annotation": 2}
