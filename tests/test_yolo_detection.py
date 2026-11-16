"""T37: the YOLO detector arm (4Y), the full detection metrics (FROC, AP75, FP per image) and test-only auto crops.

CPU, synthetic, no downloads: ultralytics is never imported (training and detection take a fake predictor).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pycocotools")

from src.data.create_auto_roi_dataset import main as crops_main  # noqa: E402
from src.detection import yolo  # noqa: E402
from src.detection.common import photo_table  # noqa: E402
from src.detection.detection_metrics import (  # noqa: E402
    evaluate_photos,
    froc_points,
    main as metrics_main,
    photo_hits,
    sweep,
    with_patients,
)
from src.detection.evaluate_detection import candidates, reference_photos  # noqa: E402
from src.detection.select import main as select_main  # noqa: E402
from tests.test_auto_roi import SIZE, frozen_selection, good_detections, select_args, world  # noqa: E402,F401
from tests.test_detection_evaluation import detections_frame, random_case, reference_frame  # noqa: E402


# ---------- YOLO arm ----------

def test_labels_are_normalised_centres_and_photos_shrink_to_the_store_side():
    lines = yolo.label_lines(np.array([[20, 20, 60, 50]], float), 200, 150)
    assert lines == ["0 0.200000 0.233333 0.200000 0.200000"]
    from PIL import Image
    small, scale = yolo.shrink(Image.new("RGB", (4000, 3000)))
    assert small.size == (1600, 1200) and scale == pytest.approx(0.4)
    same, one = yolo.shrink(Image.new("RGB", (200, 150)))
    assert same.size == (200, 150) and one == 1.0
    assert yolo.config_order() == ["yolo11s", "yolo11m", "yolo11l"]


def test_prepare_writes_train_and_validation_only(world, tmp_path):
    out = tmp_path / "yolo_data"
    args = ["prepare", "--image-root", str(world["raw"]), "--output-dir", str(out),
            "--reference", str(world["reference_csv"]), "--box-manifest", str(world["box_manifest"])]
    assert yolo.main(args) == 0
    photos = pd.read_csv(out / "photos.csv")
    # Test photos never enter the YOLO dataset
    assert set(photos["split"]) == {"train", "validation"} and len(photos) == 12
    ref = world["reference"]
    for row in photos.to_dict("records"):
        label = (out / row["image"].replace("images/", "labels/").replace(".jpg", ".txt")).read_text().split("\n")
        boxes = ref.loc[ref["relative_path"] == row["relative_path"], ["x1", "y1", "x2", "y2"]].to_numpy(float)
        assert [x for x in label if x] == yolo.label_lines(boxes, *SIZE)
        assert (out / row["image"]).is_file()
    assert "train: images/train" in (out / "data.yaml").read_text()
    # Written once, and never inside the repository
    with pytest.raises(SystemExit, match="not empty"):
        yolo.main(args)
    with pytest.raises(ValueError, match="inside the repository"):
        yolo.main([*args[:4], str(Path(__file__).resolve().parents[1] / "yolo_tmp"), *args[5:]])


def test_detections_are_scaled_back_to_source_pixels(world):
    photos = photo_table(world["box_manifest"], ("validation",))
    seen = []

    def fake(image):
        seen.append(image.size)
        return np.array([[10, 10, 30, 25]], float), np.array([0.8])

    rows = yolo.detect_photos(fake, photos, world["raw"], side=100)
    # The 200 x 150 photos were shrunk by half, so the box doubles back
    assert set(seen) == {(100, 75)} and len(rows) == len(photos)
    assert [rows[0][k] for k in ("x1", "y1", "x2", "y2", "score")] == [20, 20, 60, 50, 0.8]


def test_test_photos_need_the_frozen_selection_and_runs_are_named_by_size(world, tmp_path):
    run = tmp_path / "yolo11m"
    with pytest.raises(SystemExit, match="selection.json"):
        yolo.main(["detect", "--run-dir", str(run), "--splits", "test", "--output", str(tmp_path / "t.csv"),
                   "--image-root", str(world["raw"])])
    sel = frozen_selection(world, tmp_path / "sel.json")  # chose "good", not yolo11m
    with pytest.raises(SystemExit, match="chose"):
        yolo.main(["detect", "--run-dir", str(run), "--splits", "test", "--output", str(tmp_path / "t.csv"),
                   "--image-root", str(world["raw"]), "--selection", str(sel)])
    # --detector-dir takes the chosen run from the selection
    with pytest.raises(SystemExit, match="needs --selection"):
        yolo.main(["detect", "--detector-dir", str(tmp_path), "--splits", "test", "--image-root", str(world["raw"])])
    with pytest.raises(SystemExit):
        yolo.main(["train", "--size", "s", "--weights", "w.pt", "--data-dir", str(tmp_path),
                   "--output-dir", str(tmp_path / "yolo11m")])


def test_4y_selects_on_validation_only(world, tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    ref = world["reference"]
    good_detections(ref, ("train", "validation", "test")).to_csv(dev / "yolo11s.csv", index=False)
    assert select_main(select_args(world, dev, tmp_path / "selection.json", "yolo11s", arm="4Y")) == 0
    chosen = json.loads((tmp_path / "selection.json").read_text())
    assert chosen["selection_splits"] == ["validation"] and chosen["gate_passed"]
    assert chosen["selection_photos"] == 6 and chosen["test_detections_dropped"]["yolo11s"] > 0


def test_auto_crops_can_be_cut_for_test_only(world, tmp_path):
    ref = world["reference"]
    good_detections(ref, ("train", "validation", "test")).to_csv(tmp_path / "all.csv", index=False)
    sel = frozen_selection(world, tmp_path / "selection.json")
    out = tmp_path / "roi_auto"
    assert crops_main(["--selection", str(sel), "--detections", str(tmp_path / "all.csv"), "--splits", "test",
                       "--box-manifest", str(world["box_manifest"]), "--limits", str(world["limits"]),
                       "--output-dir", str(out), "--image-root", str(world["raw"]),
                       "--crop-root", str(tmp_path / "crops")]) == 0
    box = pd.read_csv(out / "box" / "roi_manifest.csv")
    assert set(box["split"]) == {"test"} and box["relative_path"].nunique() == 6
    assert (box["fallback"] == "none").all()
    summary = json.loads((out / "crop_summary.json").read_text())
    assert set(summary["fallback_by_split"]) == {"test"} and summary["photos"] == 6


# ---------- detection metrics ----------

def hand_case():
    photos = reference_photos(reference_frame([
        ("A", "c1", "cut wound", "test", [[0, 0, 100, 100], [500, 500, 600, 600]]),
        ("B", "c2", "scratch", "test", [[0, 0, 100, 100]]),
    ]), ["test"])
    detections = detections_frame([
        ["A", 0.9, 0, 0, 100, 100], ["A", 0.8, 300, 300, 400, 400], ["A", 0.3, 500, 500, 600, 600],
        ["B", 0.7, 700, 700, 800, 800], ["B", 0.6, 0, 0, 100, 100],
    ])
    return photos, candidates(detections, photos, 0.5)


def test_froc_curve_and_sensitivity_at_fixed_false_positive_rates():
    photos, cands = hand_case()
    curve = sweep(photo_hits(photos, cands, 35), n_true=3, n_photos=2)
    # Scores in order: 0.9 hit, 0.8 FP, 0.7 FP, 0.6 hit, 0.3 hit
    np.testing.assert_allclose(curve["sensitivity"], [0, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 1])
    np.testing.assert_allclose(curve["fp_per_image"], [0, 0, 0.5, 1, 1, 1])
    points = froc_points(curve)
    assert points["sensitivity_at_0.5fp"] == pytest.approx(1 / 3)
    assert points["sensitivity_at_1fp"] == pytest.approx(1.0)
    assert points["froc_score"] == pytest.approx(5 / 7)


def test_operating_point_metrics_by_hand():
    photos, cands = hand_case()
    result, results, _ = evaluate_photos(photos, cands, threshold=0.5, max_boxes=35, size_bins=None,
                                         n_resamples=0, seed=42)
    m = result["metrics"]
    # Final boxes: A 0.9 (hit), 0.8 (FP); B 0.7 (FP), 0.6 (hit)
    assert (m["n_predicted"], m["n_matched"], m["n_true"]) == (4, 2, 3)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(4 / 7) and m["fp_per_image"] == pytest.approx(1.0)
    # The missed box of A has no overlapping final box: best IoUs 1, 0, 1
    assert m["mean_best_iou"] == pytest.approx(2 / 3) and m["mean_iou"] == pytest.approx(1.0)
    assert m["ap50"] > m["ap75"] - 1e-9 and 0 < m["ap"] <= 1


def test_bootstrap_is_seeded_and_patients_can_be_the_unit(tmp_path):
    photos, detections = random_case(0)
    cands = candidates(detections, photos, 0.5)
    a, _, _ = evaluate_photos(photos, cands, 0.5, 35, None, n_resamples=30, seed=42)
    b, _, _ = evaluate_photos(photos, cands, 0.5, 35, None, n_resamples=30, seed=42)
    assert a["intervals"] == b["intervals"]  # (metrics hold NaN size APs, so compare the intervals)
    assert {"ap75", "fp_per_image", "sensitivity_at_1fp", "froc_score"} <= set(a["intervals"])
    assert a["intervals"]["recall"]["low"] <= a["metrics"]["recall"] <= a["intervals"]["recall"]["high"]
    # All photos of one patient: the patient is the only case
    pd.DataFrame({"relative_path": photos.index, "patient_case": "7"}).to_csv(tmp_path / "patients.csv", index=False)
    assert with_patients(photos, tmp_path / "patients.csv")["case"].nunique() == 1


def test_cli_evaluates_arms_refuses_a_failed_gate_and_reports(tmp_path):
    photos, detections = random_case(1)
    reference = reference_frame([(p, info["case"], info["label"], "test", info["truth"].tolist())
                                 for p, info in photos.iterrows()])
    reference.to_csv(tmp_path / "reference.csv", index=False)
    detections.to_csv(tmp_path / "detections.csv", index=False)
    (tmp_path / "limits.json").write_text(json.dumps({"max_box_share": 0.5, "max_boxes_per_photo": 35,
                                                      "size_bins": [0.005, 0.02]}))
    common = ["evaluate", "--detections", str(tmp_path / "detections.csv"), "--n-resamples", "5",
              "--reference", str(tmp_path / "reference.csv"), "--limits", str(tmp_path / "limits.json")]
    (tmp_path / "failed.json").write_text(json.dumps({"threshold": 0.5, "gate_passed": False}))
    with pytest.raises(SystemExit, match="recall gate"):
        metrics_main([*common, "--selection", str(tmp_path / "failed.json"), "--output-dir", str(tmp_path / "x")])
    for arm in ("4Y", "4F"):
        (tmp_path / f"{arm}.json").write_text(json.dumps({"arm": arm, "chosen_config": "c", "threshold": 0.5,
                                                          "gate_passed": True}))
        assert metrics_main([*common, "--selection", str(tmp_path / f"{arm}.json"),
                             "--output-dir", str(tmp_path / arm)]) == 0
    report = json.loads((tmp_path / "4Y" / "metrics.json").read_text())
    assert set(report["by_type"]) == {"cut wound", "scratch"} and "ap75" in report["overall"]["intervals"]
    assert metrics_main(["report", "--arm", f"4Y={tmp_path / '4Y'}", f"4F={tmp_path / '4F'}",
                         "--output-dir", str(tmp_path / "report")]) == 0
    table = pd.read_csv(tmp_path / "report" / "detection_metrics_table.csv")
    assert {"AP50", "AP75", "mAP50-95", "False positives per image", "Sensitivity at 1 FP/image"} <= set(table["metric"])
    assert (tmp_path / "report" / "froc_pr.png").is_file()
    assert set(pd.read_csv(tmp_path / "report" / "detection_by_type.csv")["wound_type"]) == {"cut wound", "scratch"}


def test_a_test_only_crop_manifest_can_be_predicted(world, tmp_path):
    from src.evaluation.predict_on_manifest import read_auto_manifest
    rows = pd.read_csv(world["box_manifest"], dtype=str)
    # Seven classes are required; give the synthetic test crops all seven labels
    test = rows[rows["split"] == "test"].head(7).copy()
    test["canonical_wound_label"] = ["contusion", "cut wound", "graze wound", "gunshot wound", "laceration",
                                     "penetration wound", "scratch"]
    test.to_csv(tmp_path / "test_only.csv", index=False)
    assert set(read_auto_manifest(tmp_path / "test_only.csv", "crop")["split"]) == {"test"}
    test.head(6).to_csv(tmp_path / "missing.csv", index=False)
    with pytest.raises(ValueError, match="Expected 7 classes"):
        read_auto_manifest(tmp_path / "missing.csv", "crop")
