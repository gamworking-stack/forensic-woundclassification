"""Unit tests for ROI preprocessing, union bounding box math, and split inheritance."""

import json
import sys
import tempfile
import zipfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PIL import Image
from src.data.audit_full_vs_roi import audit_full_vs_roi, audit_problems
from src.data.create_roi_dataset import (
    build_roi_records,
    compute_union_box,
    find_source_image,
    normalize_label,
    parse_yolo_export,
    resolve_export_files,
)


def write_yolo_zip(path: Path, names, annotations) -> Path:
    """Write a minimal CVAT YOLO 1.1 export: obj.names + obj_train_data/<stem>.txt."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("obj.names", "\n".join(names) + "\n")
        for stem, lines in annotations.items():
            archive.writestr(f"obj_train_data/{stem}.txt", "\n".join(lines) + "\n")
    return path


def test_resolve_export_files_skips_byte_identical_archives():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        original = write_yolo_zip(tmp / "task_17.zip", ["scratch"], {"IMG_1": ["0 0.5 0.5 0.2 0.2"]})
        copy = tmp / "task_17 (1).zip"
        copy.write_bytes(original.read_bytes())
        other = write_yolo_zip(tmp / "task_18.zip", ["cut"], {"IMG_2": ["0 0.5 0.5 0.2 0.2"]})

        resolved = resolve_export_files([original, copy, other])

        assert resolved == [original, other], resolved


def test_normalize_label():
    assert normalize_label("cut") == "cut wound"
    assert normalize_label("graze") == "graze wound"
    assert normalize_label("gsw") == "gunshot wound"
    assert normalize_label("penetrating wound (sharp&blunt)") == "penetration wound"
    assert normalize_label("contusion") == "contusion"
    assert normalize_label("laceration") == "laceration"
    assert normalize_label("scratch") == "scratch"


def test_compute_union_box_single_box():
    # Box: center=(0.5, 0.5), w=0.2, h=0.2 on 1000x1000 image
    # x1=400, y1=400, x2=600, y2=600.
    # pad=0.08: delta_x = 200 * 0.08 = 16.
    # Expected: x1=384, y1=384, x2=616, y2=616.
    box = [[0.5, 0.5, 0.2, 0.2]]
    x1, y1, x2, y2 = compute_union_box(box, width=1000, height=1000, padding=0.08)
    assert x1 == 384
    assert y1 == 384
    assert x2 == 616
    assert y2 == 616


def test_compute_union_box_multiple_boxes():
    # Box 1: (0.2, 0.2, 0.1, 0.1) -> x: [150, 250], y: [150, 250]
    # Box 2: (0.7, 0.7, 0.2, 0.2) -> x: [600, 800], y: [600, 800]
    # Union unpadded: x: [150, 800], y: [150, 800] -> span = 650
    # pad = 650 * 0.08 = 52
    # Expected: x1 = 150 - 52 = 98, x2 = 800 + 52 = 852
    boxes = [
        [0.2, 0.2, 0.1, 0.1],
        [0.7, 0.7, 0.2, 0.2],
    ]
    x1, y1, x2, y2 = compute_union_box(boxes, width=1000, height=1000, padding=0.08)
    assert x1 == 98
    assert y1 == 98
    assert x2 == 852
    assert y2 == 852


def test_compute_union_box_boundary_clamping():
    # Box touching image border
    boxes = [[0.05, 0.05, 0.1, 0.1]]  # x: [0, 100], y: [0, 100]
    x1, y1, x2, y2 = compute_union_box(boxes, width=1000, height=1000, padding=0.08)
    assert x1 == 0  # clamped to 0
    assert y1 == 0  # clamped to 0
    assert x2 == 108
    assert y2 == 108


def test_resolve_export_files_prefers_original_name_over_browser_copy():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        original = write_yolo_zip(tmp / "task_17 1.1.zip", ["scratch"], {"IMG_1": ["0 0.5 0.5 0.2 0.2"]})
        copy = tmp / "task_17 1.1 (1).zip"
        copy.write_bytes(original.read_bytes())

        # Shell globs sort "1.1 (1).zip" before "1.1.zip" because space < "."
        assert resolve_export_files([copy, original]) == [original]


def test_find_source_image_resolves_raw_label_folder_and_missing_extension():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "cut").mkdir()
        (root / "cut" / "IMG_0235.JPG").write_bytes(b"jpeg")
        (root / "gsw").mkdir()
        (root / "gsw" / "P1010419.JPG").write_bytes(b"jpeg")

        no_extension = {"filename": "IMG_0235", "wound_label": "cut", "canonical_wound_label": "cut wound"}
        with_extension = {"filename": "P1010419.JPG", "wound_label": "gsw", "canonical_wound_label": "gunshot wound"}
        absent = {"filename": "IMG_9999", "wound_label": "cut", "canonical_wound_label": "cut wound"}

        assert find_source_image(root, no_extension) == root / "cut" / "IMG_0235.JPG"
        assert find_source_image(root, with_extension) == root / "gsw" / "P1010419.JPG"
        assert find_source_image(root, absent) is None


def manifest_row(case_id, filename, wound_label, canonical, split):
    return {
        "pseudonimized_id": case_id,
        "wound_id": "1.0",
        "filename": filename,
        "wound_label": wound_label,
        "canonical_wound_label": canonical,
        "relative_path": f"{canonical}/{filename}",
        "split": split,
    }


SYNTHETIC_MANIFEST = [
    manifest_row("1", "IMG_A", "cut", "cut wound", "train"),
    manifest_row("2", "IMG_B.JPG", "scratch", "scratch", "test"),
]


def test_build_roi_records_dry_run_inherits_split_and_reports_unannotated():
    with tempfile.TemporaryDirectory() as tmp:
        export = write_yolo_zip(
            Path(tmp) / "task_1.zip",
            ["cut", "scratch"],
            {"IMG_A": ["0 0.5 0.5 0.2 0.2"], "IMG_Z": ["1 0.5 0.5 0.1 0.1"]},
        )

        result = build_roi_records(
            SYNTHETIC_MANIFEST,
            [export],
            dimensions={("cut wound", "IMG_A"): (1000, 500)},
        )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert (row["pseudonimized_id"], row["split"], row["canonical_wound_label"]) == ("1", "train", "cut wound")
    # x: 400..600 padded by 16 px; y: 200..300 padded by 8 px
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (384, 192, 616, 308)
    assert (row["source_width"], row["source_height"]) == (1000, 500)
    assert row["roi_relative_path"] == "train/cut wound/IMG_A_roi.jpg"

    assert [(d["pseudonimized_id"], d["filename"], d["split"], d["reason"]) for d in result.dropped] == [
        ("2", "IMG_B.JPG", "test", "no_annotation")
    ]
    assert [(u["stem"], u["label"]) for u in result.unmatched_annotations] == [("IMG_Z", "scratch")]


def test_build_roi_records_live_mode_crops_exif_oriented_image_and_drops_missing_source():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image_root, output_dir = tmp / "raw", tmp / "roi"
        (image_root / "cut").mkdir(parents=True)
        # Stored portrait 500x1000 with EXIF orientation 6 -> displayed (and annotated) as 1000x500.
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.new("RGB", (500, 1000), "red").save(image_root / "cut" / "IMG_A.JPG", exif=exif)
        export = write_yolo_zip(
            tmp / "task_1.zip",
            ["cut", "scratch"],
            {"IMG_A": ["0 0.5 0.5 0.2 0.2"], "IMG_B": ["1 0.5 0.5 0.2 0.2"]},
        )

        result = build_roi_records(SYNTHETIC_MANIFEST, [export], image_root=image_root, output_dir=output_dir)

        assert len(result.rows) == 1
        row = result.rows[0]
        assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (384, 192, 616, 308)
        assert (row["source_width"], row["source_height"]) == (1000, 500)
        assert row["source_relative_path"] == "cut/IMG_A.JPG"
        with Image.open(output_dir / "train" / "cut wound" / "IMG_A_roi.jpg") as crop:
            assert crop.size == (232, 116)
        assert [(d["filename"], d["reason"]) for d in result.dropped] == [("IMG_B.JPG", "source_image_missing")]


def test_build_roi_records_rejects_same_image_annotated_in_two_distinct_exports():
    with tempfile.TemporaryDirectory() as tmp:
        first = write_yolo_zip(Path(tmp) / "task_1.zip", ["cut"], {"IMG_A": ["0 0.5 0.5 0.2 0.2"]})
        second = write_yolo_zip(Path(tmp) / "task_2.zip", ["cut"], {"IMG_A": ["0 0.4 0.4 0.2 0.2"]})
        try:
            build_roi_records(SYNTHETIC_MANIFEST, [first, second], dimensions={("cut wound", "IMG_A"): (100, 100)})
        except ValueError as error:
            assert "collision" in str(error).lower()
        else:
            raise AssertionError("Expected an annotation collision error")


def test_parse_yolo_export_normalizes_labels_and_skips_invalid_lines():
    with tempfile.TemporaryDirectory() as tmp:
        export = write_yolo_zip(
            Path(tmp) / "task_1.zip",
            ["gsw", "Penetrating wound (sharp&blunt)"],
            {
                "IMG_A": [
                    "0 0.5 0.5 0.2 0.2",
                    "1 0.3 0.3 0.1 0.1",
                    "2 0.5 0.5 0.2 0.2",   # class id out of range
                    "0 0.5 0.5 1.2 0.2",   # width outside [0, 1]
                    "0 0.5 0.5 0.2",       # too few fields
                ],
            },
        )
        labels, annotations, invalid_lines = parse_yolo_export(export)

    assert labels == ["gunshot wound", "penetration wound"]
    assert annotations["IMG_A"] == [
        ("gunshot wound", [0.5, 0.5, 0.2, 0.2]),
        ("penetration wound", [0.3, 0.3, 0.1, 0.1]),
    ]
    assert [(i["stem"], i["line_number"], i["reason"]) for i in invalid_lines] == [
        ("IMG_A", 3, "class_id_out_of_range"),
        ("IMG_A", 4, "box_out_of_bounds"),
        ("IMG_A", 5, "wrong_field_count"),
    ]


AUDIT_FULL = pd.DataFrame([
    manifest_row("1", "A.JPG", "cut", "cut wound", "train"),
    manifest_row("1", "B.JPG", "cut", "cut wound", "train"),
    manifest_row("2", "C", "scratch", "scratch", "test"),
    manifest_row("3", "D.JPG", "scratch", "scratch", "validation"),
])


def roi_rows(*filenames):
    return AUDIT_FULL[AUDIT_FULL["filename"].isin(filenames)].copy()


def test_audit_identifies_full_images_missing_from_roi():
    audit = audit_full_vs_roi(AUDIT_FULL, roi_rows("A.JPG", "B.JPG", "D.JPG"))

    assert [(m["pseudonimized_id"], m["filename"], m["split"]) for m in audit.missing_from_roi] == [
        ("2", "C", "test")
    ]
    # Case 2 had only this image, so the whole case leaves the ROI path.
    assert audit.missing_from_roi[0]["case_lost_entirely"] is True

    partial = audit_full_vs_roi(AUDIT_FULL, roi_rows("A.JPG", "C", "D.JPG"))
    assert [(m["filename"], m["case_lost_entirely"]) for m in partial.missing_from_roi] == [("B.JPG", False)]


def test_audit_explains_each_missing_image_with_its_crop_drop_reason():
    drop_reasons = {("scratch", "C"): "no_annotation"}
    audit = audit_full_vs_roi(AUDIT_FULL, roi_rows("A.JPG", "D.JPG"), drop_reasons=drop_reasons)

    assert sorted((m["filename"], m["reason"]) for m in audit.missing_from_roi) == [
        ("B.JPG", "unexplained"),
        ("C", "no_annotation"),
    ]


def test_audit_problems_flags_unexplained_losses_but_accepts_explained_ones():
    roi = roi_rows("A.JPG", "B.JPG", "D.JPG")
    explained = audit_full_vs_roi(AUDIT_FULL, roi, drop_reasons={("scratch", "C"): "no_annotation"})
    unexplained = audit_full_vs_roi(AUDIT_FULL, roi)

    assert not any("drop reason" in p for p in audit_problems(explained))
    assert any("drop reason" in p and "C" in p for p in audit_problems(unexplained))


def test_audit_rejects_manifest_with_unknown_split():
    bad_roi = roi_rows("A.JPG")
    bad_roi["split"] = "holdout"
    try:
        audit_full_vs_roi(AUDIT_FULL, bad_roi)
    except ValueError as error:
        assert "holdout" in str(error)
    else:
        raise AssertionError("Expected ValueError for unknown split")


def test_audit_table_compares_images_and_cases_per_split_and_class():
    table = audit_full_vs_roi(AUDIT_FULL, roi_rows("A.JPG", "B.JPG", "D.JPG")).table
    rows = {(r["split"], r["canonical_wound_label"]): r for r in table.to_dict("records")}
    columns = ["full_images", "roi_images", "image_retention_pct", "full_cases", "roi_cases", "case_retention_pct"]

    def counts(split, label):
        return tuple(rows[(split, label)][c] for c in columns)

    assert counts("train", "cut wound") == (2, 2, 100.0, 1, 1, 100.0)
    assert counts("test", "scratch") == (1, 0, 0.0, 1, 0, 0.0)
    assert counts("all", "scratch") == (2, 1, 50.0, 2, 1, 50.0)
    assert counts("all", "all") == (4, 3, 75.0, 3, 2, 66.67)
    # Classes absent from a split still get a row so the table is a complete grid.
    assert counts("validation", "cut wound")[:2] == (0, 0)
    assert pd.isna(rows[("validation", "cut wound")]["image_retention_pct"])


def test_audit_reports_roi_rows_that_break_split_inheritance():
    roi = roi_rows("A.JPG", "B.JPG", "D.JPG")
    roi.loc[roi["filename"] == "D.JPG", "split"] = "train"
    roi.loc[roi["filename"] == "B.JPG", "pseudonimized_id"] = "9"
    stray = pd.DataFrame([manifest_row("4", "E.JPG", "cut", "cut wound", "test")])

    audit = audit_full_vs_roi(AUDIT_FULL, pd.concat([roi, stray], ignore_index=True))

    assert sorted((d["filename"], d["type"]) for d in audit.discrepancies) == [
        ("B.JPG", "case_mismatch"),
        ("D.JPG", "split_mismatch"),
        ("E.JPG", "not_in_full_manifest"),
    ]


def test_audit_has_no_discrepancies_for_faithful_roi_subset():
    assert audit_full_vs_roi(AUDIT_FULL, roi_rows("A.JPG", "B.JPG", "D.JPG")).discrepancies == []


def test_audit_coverage_reports_missing_classes_and_case_leakage_per_path():
    leaky_roi = roi_rows("A.JPG", "B.JPG", "D.JPG")
    leaky_roi.loc[leaky_roi["filename"] == "B.JPG", "split"] = "test"

    coverage = audit_full_vs_roi(AUDIT_FULL, leaky_roi).coverage

    assert coverage["full"]["leaked_cases"] == []
    assert coverage["full"]["missing_classes"] == {
        "train": ["scratch"], "validation": ["cut wound"], "test": ["cut wound"],
    }
    assert coverage["roi"]["leaked_cases"] == ["1"]
    assert coverage["roi"]["missing_classes"]["test"] == ["scratch"]


def test_roi_manifest_integrity():
    manifest_path = Path("data/roi/roi_manifest.csv")
    assert manifest_path.exists(), "data/roi/roi_manifest.csv must exist"

    df = pd.read_csv(manifest_path, dtype=str)
    assert len(df) == 1100, f"Expected 1100 rows in ROI manifest, got {len(df)}"
    # 421 source cases minus 3 single-image cases whose only image is unannotated (220, 353, 415).
    assert df["pseudonimized_id"].nunique() == 418

    # Every ROI row inherits case and split from exactly one source manifest row.
    source = pd.read_csv("data/splits/case_level_split_manifest.csv", dtype=str)
    joined = df.merge(source, on=["filename", "canonical_wound_label"], suffixes=("", "_source"), validate="one_to_one")
    assert len(joined) == 1100
    assert (joined["pseudonimized_id"] == joined["pseudonimized_id_source"]).all()
    assert (joined["split"] == joined["split_source"]).all()

    # Check zero case leakage
    train_cases = set(df[df["split"] == "train"]["pseudonimized_id"])
    val_cases = set(df[df["split"] == "validation"]["pseudonimized_id"])
    test_cases = set(df[df["split"] == "test"]["pseudonimized_id"])

    assert train_cases.isdisjoint(val_cases), "Train & Validation case leakage in ROI!"
    assert train_cases.isdisjoint(test_cases), "Train & Test case leakage in ROI!"
    assert val_cases.isdisjoint(test_cases), "Validation & Test case leakage in ROI!"

    # Check all 7 classes in every split
    for split_name in ("train", "validation", "test"):
        classes = df[df["split"] == split_name]["canonical_wound_label"].unique()
        assert len(classes) == 7, f"Split '{split_name}' missing classes: {classes}"


def test_crop_summary_json():
    summary_path = Path("data/roi/crop_summary.json")
    assert summary_path.exists(), "data/roi/crop_summary.json must exist"

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["counts"]["manifest_rows"] == 1106
    assert summary["counts"]["crops_created"] == 1100
    assert summary["counts"]["manifest_rows_without_usable_annotation"] == 6
    assert summary["unannotated_records_count"] == 6
    assert summary["crops_by_split"]["train"] == 779
    assert summary["crops_by_split"]["validation"] == 164
    assert summary["crops_by_split"]["test"] == 157
    assert summary["unique_cases"] == 418
    assert summary["skipped_records"] == []
    assert summary["invalid_annotation_lines"] == []
    for split_name in ("train", "validation", "test"):
        assert len(summary["crops_by_split_and_class"][split_name]) == 7, split_name
    assert len(summary["annotation_exports"]) == 7

    unannotated = {(r["pseudonimized_id"], r["filename"], r["split"]) for r in summary["unannotated_records"]}
    assert unannotated == {
        ("94", "ZZ140321.JPG", "validation"),
        ("167", "ZZ310855.JPG", "train"),
        ("220", "PC235480.JPG", "validation"),
        ("353", "P1070307.JPG", "test"),
        ("388", "IMG_3520.JPG", "train"),
        ("415", "P1071425", "train"),
    }


def test_full_vs_roi_audit_report():
    report_path = Path("data/audits/full_vs_roi_audit_report.json")
    assert report_path.exists(), "data/audits/full_vs_roi_audit_report.json must exist"
    with open(report_path) as f:
        report = json.load(f)

    missing = {(m["pseudonimized_id"], m["filename"], m["canonical_wound_label"], m["split"])
               for m in report["missing_from_roi"]}
    assert missing == {
        ("94", "ZZ140321.JPG", "scratch", "validation"),
        ("167", "ZZ310855.JPG", "contusion", "train"),
        ("220", "PC235480.JPG", "scratch", "validation"),
        ("353", "P1070307.JPG", "scratch", "test"),
        ("388", "IMG_3520.JPG", "scratch", "train"),
        ("415", "P1071425", "cut wound", "train"),
    }
    assert report["audit_passed"] is True and report["problems"] == []
    assert {m["reason"] for m in report["missing_from_roi"]} == {"no_annotation"}
    assert report["discrepancies"] == []
    assert report["images_by_split"]["test"] == {"full": 158, "roi": 157}

    for path_name in ("full", "roi"):
        coverage = report["coverage"][path_name]
        assert coverage["leaked_cases"] == [], path_name
        assert all(not classes for classes in coverage["missing_classes"].values()), path_name

    # Retention rates as stated in specs/phase2_roi_preprocessing.md section 3.2.
    expected_retention = {
        "gunshot wound": 100.0, "graze wound": 100.0, "laceration": 100.0, "contusion": 99.5,
        "cut wound": 98.9, "penetration wound": 100.0, "scratch": 94.2,
    }
    observed = {label: round(r["image_retention_pct"], 1) for label, r in report["retention_by_class"].items()}
    assert observed == expected_retention


def test_full_vs_roi_audit_table_matches_manifests():
    full = pd.read_csv("data/splits/case_level_split_manifest.csv", dtype=str)
    roi = pd.read_csv("data/roi/roi_manifest.csv", dtype=str)
    table = pd.read_csv("data/audits/full_vs_roi_audit_table.csv")

    overall = table[(table["split"] == "all") & (table["canonical_wound_label"] == "all")].iloc[0]
    assert (overall["full_images"], overall["roi_images"]) == (len(full), len(roi))
    assert (overall["full_cases"], overall["roi_cases"]) == (
        full["pseudonimized_id"].nunique(), roi["pseudonimized_id"].nunique()
    )

    grid = table[(table["split"] != "all") & (table["canonical_wound_label"] != "all")]
    for path, df in (("full", full), ("roi", roi)):
        expected = df.groupby(["split", "canonical_wound_label"]).size()
        observed = grid.set_index(["split", "canonical_wound_label"])[f"{path}_images"]
        assert observed[observed > 0].sort_index().equals(expected.sort_index()), path


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} ROI preprocessing tests passed.")
