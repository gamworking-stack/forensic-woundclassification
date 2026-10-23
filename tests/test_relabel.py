"""T23: the 23 Oct 2026 relabelled metadata and annotations, photos relabelled after annotation, WOUND_DATA_DIR."""

import importlib
import sys
import tempfile
import zipfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.data.create_roi_dataset import build_roi_records, find_source_image, manifest_key, parse_export
from src.data.prepare_relabel import case_groups, match_reference, prepare
from tests.test_roi_preprocessing import manifest_row, write_yolo_zip


RELABEL_DIR = Path("data/relabel_2026_10_23")


def sheet(rows):
    # Sheet rows: (case, patient, filename, raw label, inclusion)
    return pd.DataFrame(rows, columns=["pseudonimized_id", "patient_id", "filename", "wound_label", "Inclusion"]).assign(
        wound_id="1")


def test_case_groups_join_cases_sharing_a_patient_but_not_a_placeholder():
    df = sheet([
        ("1", "A", "a.jpg", "cut", "TRUE"),
        ("2", "A", "b.jpg", "cut", "TRUE"),   # same patient as case 1
        ("3", "B", "c.jpg", "cut", "TRUE"),
        ("3", "C", "d.jpg", "cut", "TRUE"),   # case 3 holds patients B and C
        ("4", "C", "e.jpg", "cut", "TRUE"),   # ... so case 4 joins case 3
        ("5", "n/a", "f.jpg", "cut", "TRUE"),
        ("6", "n/a", "g.jpg", "cut", "TRUE"),  # a placeholder joins nothing
    ])
    # Each group is named by its smallest case ID
    assert case_groups(df).tolist() == [1, 1, 3, 3, 3, 5, 6]


def test_prepare_excludes_rows_keeps_source_label_and_lists_relabelled_photos():
    df = sheet([
        ("1", "A", "a.jpg", "gsw", "TRUE"),
        ("1", "A", "b.jpg", "penetrating wound (sharp&blunt)", "TRUE"),  # was gsw when annotated
        ("2", "B", "c.jpg", "graze", "FALSE"),
        ("3", "C", "d.jpg", "imprint", "TRUE"),
    ])
    reference = pd.DataFrame({
        "pseudonimized_id": ["1", "1", "2"], "filename": ["a.jpg", "b.jpg", "c.jpg"],
        "wound_label": ["gsw", "gsw", "graze"],
    })

    kept, excluded, report = prepare(df, reference, ["imprint"])

    # Inclusion FALSE and the label outside the 7 classes are excluded, with their reasons
    assert excluded[["filename", "reason"]].values.tolist() == [
        ["c.jpg", "inclusion_false"], ["d.jpg", "label_outside_7_classes"]]
    # The relabelled photo is still filed under gsw, and is listed as relabelled
    assert kept[["filename", "wound_label", "source_label"]].values.tolist() == [
        ["a.jpg", "gsw", "gsw"], ["b.jpg", "penetrating wound (sharp&blunt)", "gsw"]]
    assert [(r["filename"], r["from"], r["to"]) for r in report["relabelled_photos"]] == [
        ("b.jpg", "gsw", "penetrating wound (sharp&blunt)")]
    assert report["included_photos"] == 2


def test_match_reference_refuses_a_photo_it_cannot_place():
    # The label changed and the filename is in the reference twice: the photo's folder is unknown
    df = sheet([("1", "A", "a.jpg", "cut", "TRUE")]).assign(source_pseudonimized_id="1")
    reference = pd.DataFrame({"pseudonimized_id": ["1", "2"], "filename": ["a.jpg", "a.jpg"],
                              "wound_label": ["gsw", "graze"]})
    with pytest.raises(ValueError, match="do not match one reference row"):
        match_reference(df, reference)


def relabelled_row():
    # A photo annotated and filed as gsw, now labelled penetration wound
    row = manifest_row("1", "IMG_A.JPG", "penetrating wound (sharp&blunt)", "penetration wound", "train")
    row["source_label"] = "gsw"
    return row


def test_relabelled_photo_is_found_and_cropped_under_its_old_label_but_keeps_its_new_class():
    row = relabelled_row()
    # The annotation key and the raw folder use the old label
    assert manifest_key(row) == ("gunshot wound", "IMG_A")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "gsw").mkdir()
        (root / "gsw" / "IMG_A.JPG").write_bytes(b"jpeg")
        assert find_source_image(root, row) == root / "gsw" / "IMG_A.JPG"

        # Its box sits in the gsw CVAT task; the crop carries the new class in its row and path
        export = write_yolo_zip(root / "task_1.zip", ["gsw"], {"IMG_A": ["0 0.5 0.5 0.2 0.2"]})
        result = build_roi_records([row], [export], dimensions={("gunshot wound", "IMG_A"): (1000, 500)})

    assert result.dropped == []
    assert result.rows[0]["canonical_wound_label"] == "penetration wound"
    assert result.rows[0]["roi_relative_path"] == "train/penetration wound/IMG_A_roi.jpg"


def test_wound_data_dir_selects_the_manifest_folder(monkeypatch):
    import src.data.paths as paths
    # Default: the original data folder
    monkeypatch.delenv("WOUND_DATA_DIR", raising=False)
    assert importlib.reload(paths).DATA_DIR == Path("data")
    # Set: every default reads the other dataset version
    monkeypatch.setenv("WOUND_DATA_DIR", str(RELABEL_DIR))
    assert importlib.reload(paths).DATA_DIR == RELABEL_DIR
    monkeypatch.delenv("WOUND_DATA_DIR")
    importlib.reload(paths)


def test_cvat_xml_export_gives_yolo_boxes_and_encloses_rotated_ones():
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations><meta><job><labels><label><name>laceration</name></label></labels></job></meta>
  <image id="0" name="IMG_A.JPG" width="1000" height="500">
    <box label="laceration" xtl="400" ytl="200" xbr="600" ybr="300" z_order="0"></box>
    <box label="laceration" xtl="400" ytl="200" xbr="600" ybr="300" rotation="90.0" z_order="0"></box>
    <polygon label="laceration" points="1,1;2,2;3,1" z_order="0"></polygon>
  </image>
</annotations>"""
    with tempfile.TemporaryDirectory() as tmp:
        export = Path(tmp) / "laceration.zip"
        with zipfile.ZipFile(export, "w") as archive:
            archive.writestr("annotations.xml", xml)
        labels, annotations, invalid = parse_export(export)

    assert labels == ["laceration"]
    (_, box), (_, rotated) = annotations["IMG_A"]
    # Pixel box 400..600 x 200..300 in a 1000 x 500 photo, as YOLO centre and size
    assert box == pytest.approx([0.5, 0.5, 0.2, 0.2])
    # Turned by 90 degrees about its centre: 100 px wide and 200 px tall
    assert rotated == pytest.approx([0.5, 0.5, 0.1, 0.4])
    # Only boxes are used
    assert [line["reason"] for line in invalid] == ["not_a_box"]


def test_committed_relabel_manifests():
    metadata = pd.read_csv(RELABEL_DIR / "validated_metadata.csv", dtype=str)
    split = pd.read_csv(RELABEL_DIR / "splits/case_level_split_manifest.csv", dtype=str)
    excluded = pd.read_csv(RELABEL_DIR / "excluded_photos.csv", dtype=str)
    # No patient identifier or demographic column is committed
    assert "patient_id" not in metadata.columns and "patient_id" not in split.columns
    # 1,113 sheet photos = 1,041 included + 65 Inclusion FALSE + 7 imprint
    assert len(metadata) == len(split) == 1041
    assert excluded["reason"].value_counts().to_dict() == {"inclusion_false": 65, "label_outside_7_classes": 7}
    # No group in two splits, and all 7 classes in every split
    assert split.groupby("pseudonimized_id")["split"].nunique().max() == 1
    assert (split.groupby("split")["canonical_wound_label"].nunique() == 7).all()

    # The crops come from the 23 Oct 2026 exports: the per-box crops are the T19 reference boxes,
    # which boxes.py checked box by box against the CVAT files
    union = pd.read_csv(RELABEL_DIR / "roi/roi_manifest.csv", dtype=str)
    per_box = pd.read_csv(RELABEL_DIR / "roi_box/roi_manifest.csv", dtype=str)
    reference = pd.read_csv(RELABEL_DIR / "roi_auto/reference_boxes.csv", dtype=str)
    assert len(union) == 1032 and len(per_box) == len(reference) == 1808
    assert set(per_box["relative_path"]) == set(union["relative_path"]) <= set(split["relative_path"])
    assert set(union["annotation_export"]) <= {f"{c}_22_Oct.zip" for c in (
        "contusion", "cut", "graze", "gsw", "laceration", "penetrate", "scratch")}
    # The two photos relabelled after annotation keep their gsw boxes but carry the new class
    relabelled = union[union["filename"].isin(["PB098804.JPG", "PB098817.JPG"])]
    assert relabelled["canonical_wound_label"].tolist() == ["penetration wound"] * 2
    assert relabelled["annotation_export"].tolist() == ["gsw_22_Oct.zip"] * 2
