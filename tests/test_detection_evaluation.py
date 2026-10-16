"""T19 detection metrics: COCO AP, the F1 threshold, the operating point, the case bootstrap and the CLI."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pycocotools")

from src.detection.boxes import final_boxes, greedy_match  # noqa: E402
from src.detection.evaluate_detection import (  # noqa: E402
    candidates,
    choose_threshold,
    coco_ap,
    finalize,
    gate_passed,
    main as evaluate_main,
    operating_metrics,
    photo_results,
    reference_photos,
    bootstrap,
)


def reference_frame(photos):
    """``photos`` = [(photo, case, label, split, [boxes])] on 1000 x 1000 images."""
    rows = [{"relative_path": p, "pseudonimized_id": c, "split": s, "canonical_wound_label": l, "box_index": k,
             "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "source_width": 1000, "source_height": 1000}
            for p, c, l, s, boxes in photos for k, b in enumerate(boxes, start=1)]
    return pd.DataFrame(rows)


def detections_frame(rows):
    return pd.DataFrame(rows, columns=["relative_path", "score", "x1", "y1", "x2", "y2"])


def test_coco_ap_matches_hand_computed_values():
    truth = np.array([[0, 0, 100, 100], [500, 500, 600, 600]], dtype=float)
    perfect = coco_ap([(truth, truth, np.array([0.9, 0.8]), 1000, 1000)])
    assert perfect["ap50"] == pytest.approx(1.0) and perfect["ap"] == pytest.approx(1.0)
    # TP 0.9, FP 0.8, TP 0.7: precision 1 up to recall 0.5, then 2/3; 101-point interpolation
    boxes = np.array([[0, 0, 100, 100], [300, 300, 400, 400], [500, 500, 600, 600]], dtype=float)
    mixed = coco_ap([(truth, boxes, np.array([0.9, 0.8, 0.7]), 1000, 1000)])
    assert mixed["ap50"] == pytest.approx((51 + 50 * 2 / 3) / 101)
    empty = coco_ap([(truth, np.zeros((0, 4)), np.zeros(0), 1000, 1000)])
    assert empty["ap50"] == 0.0 and np.isnan(empty["ap_small"])


def test_coco_ap_size_bins_use_image_area_shares():
    small, large = [0, 0, 50, 50], [0, 0, 500, 500]  # shares 0.0025 and 0.25
    truth = np.array([small, large], dtype=float)
    found_small_only = coco_ap([(truth, np.array([small], dtype=float), np.array([0.9]), 1000, 1000)],
                               size_bins=[0.01, 0.1])
    assert found_small_only["ap_small"] == pytest.approx(1.0)
    assert found_small_only["ap_large"] == pytest.approx(0.0)
    assert np.isnan(found_small_only["ap_medium"])  # no reference box in that bin


def random_case(seed):
    rng = np.random.default_rng(seed)
    photos, detections = [], []
    for i in range(12):
        truth = []
        for _ in range(int(rng.integers(1, 4))):
            x, y = rng.uniform(0, 800, 2)
            truth.append([x, y, x + rng.uniform(20, 150), y + rng.uniform(20, 150)])
        photos.append((f"p{i}", f"c{i // 2}", "cut wound" if i % 3 else "scratch", "validation", truth))
        for box in truth:
            if rng.random() < 0.7:
                jitter = rng.normal(0, 8, 4)
                detections.append([f"p{i}", round(rng.uniform(0.3, 1.0), 2), *(np.array(box) + jitter)])
        for _ in range(int(rng.integers(0, 4))):
            x, y = rng.uniform(0, 900, 2)
            detections.append([f"p{i}", round(rng.uniform(0.0, 0.8), 2), x, y, x + 60, y + 60])
    reference = reference_frame(photos)
    for row in detections:  # keep boxes ordered after jitter
        row[2], row[4] = min(row[2], row[4]), max(row[2], row[4])
        row[3], row[5] = min(row[3], row[5]), max(row[3], row[5])
    return reference_photos(reference, ["validation"]), detections_frame(detections)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_threshold_sweep_equals_brute_force(seed):
    photos, detections = random_case(seed)
    cands = candidates(detections, photos, max_share=0.5)
    chosen = choose_threshold(photos, cands, max_boxes=2)
    n_true = sum(len(t) for t in photos["truth"])
    best = None
    for t in sorted({s for _, sc in cands.values() for s in sc}, reverse=True):
        matched = predicted = 0
        for photo, info in photos.iterrows():
            boxes, scores = cands[photo]
            keep = [i for i in np.argsort(-scores, kind="stable") if scores[i] >= t][:2]
            matched += len(greedy_match(boxes[keep], scores[keep], info["truth"]))
            predicted += len(keep)
        f1 = 2 * matched / (predicted + n_true)
        if best is None or f1 > best[1] + 1e-12:  # strict: ties stay with the higher threshold
            best = (t, f1, matched / n_true)
    assert chosen["threshold"] == pytest.approx(best[0])
    assert chosen["f1"] == pytest.approx(best[1]) and chosen["recall"] == pytest.approx(best[2])


def test_threshold_ties_go_to_the_higher_score():
    photos = reference_photos(reference_frame([("p", "c", "cut wound", "validation", [[0, 0, 10, 10]])]), ["validation"])
    # One hit at 0.9; a second box at 0.5 is a miss: F1 1.0 at 0.9 beats 2/3 at 0.5
    detections = detections_frame([["p", 0.9, 0, 0, 10, 10], ["p", 0.5, 500, 500, 510, 510]])
    assert choose_threshold(photos, candidates(detections, photos, 0.5), 5)["threshold"] == 0.9
    # A real tie: 3 manual boxes; hit 0.9, misses 0.8-0.6, hit 0.5. F1 is 2/4 at 0.9 and 4/8 at 0.5
    three = reference_photos(reference_frame([("t", "c", "cut wound", "validation",
                                               [[0, 0, 10, 10], [100, 100, 110, 110], [300, 300, 310, 310]])]), ["validation"])
    det = detections_frame([["t", 0.9, 0, 0, 10, 10], ["t", 0.8, 500, 500, 510, 510], ["t", 0.7, 600, 600, 610, 610],
                            ["t", 0.6, 700, 700, 710, 710], ["t", 0.5, 100, 100, 110, 110]])
    tied = choose_threshold(three, candidates(det, three, 0.5), 35)
    assert tied["threshold"] == 0.9 and tied["f1"] == pytest.approx(0.5)
    two = reference_photos(reference_frame([("a", "c1", "cut wound", "validation", [[0, 0, 10, 10]]),
                                            ("b", "c2", "cut wound", "validation", [[0, 0, 10, 10]])]), ["validation"])
    no_boxes = choose_threshold(two, candidates(detections_frame([]), two, 0.5), 5)
    assert np.isnan(no_boxes["threshold"]) and no_boxes["recall"] == 0.0


def test_gate_boundary():
    assert gate_passed(0.50) and not gate_passed(0.4999)


def test_operating_point_counts_fallbacks_and_photo_rates():
    photos = reference_photos(reference_frame([
        ("hit", "c1", "cut wound", "test", [[0, 0, 100, 100], [500, 500, 600, 600]]),
        ("low", "c2", "scratch", "test", [[0, 0, 100, 100]]),
        ("none", "c3", "scratch", "test", [[0, 0, 100, 100]]),
    ]), ["test"])
    detections = detections_frame([["hit", 0.9, 0, 0, 100, 100], ["low", 0.2, 0, 0, 100, 100]])
    cands = candidates(detections, photos, 0.5)
    results = photo_results(photos, finalize(photos, cands, threshold=0.5, max_boxes=35))
    assert results.set_index("relative_path")["fallback"].to_dict() == {"hit": "none", "low": "top_box", "none": "full_image"}
    m = operating_metrics(results)
    assert (m["n_true"], m["n_predicted"], m["n_matched"]) == (4, 3, 2)  # the full-image box is a prediction
    assert m["recall"] == pytest.approx(0.5) and m["precision"] == pytest.approx(2 / 3)
    assert m["share_no_box"] == pytest.approx(2 / 3) and m["share_full_image"] == pytest.approx(1 / 3)
    assert m["photo_any_found"] == pytest.approx(2 / 3) and m["photo_all_found"] == pytest.approx(1 / 3)
    assert m["photo_mean_recall"] == pytest.approx((0.5 + 1 + 0) / 3) and m["mean_iou"] == pytest.approx(1.0)


def test_bootstrap_is_seeded_and_keeps_cases_together():
    photos, detections = random_case(0)
    cands = candidates(detections, photos, 0.5)
    results = photo_results(photos, finalize(photos, cands, 0.5, 35))
    a = bootstrap(photos, cands, results, n_resamples=40, seed=42)
    b = bootstrap(photos, cands, results, n_resamples=40, seed=42)
    assert a == b and set(a) == {"ap50", "precision", "recall", "photo_any_found", "photo_all_found"}
    point = operating_metrics(results)["recall"]
    assert a["recall"]["low"] <= point <= a["recall"]["high"]


def test_cli_writes_reports_and_refuses_test_for_a_failed_arm(tmp_path):
    photos, detections = random_case(1)
    reference = reference_frame([(p, info["case"], info["label"], "test", info["truth"].tolist())
                                 for p, info in photos.iterrows()])
    reference.to_csv(tmp_path / "reference.csv", index=False)
    detections.to_csv(tmp_path / "detections.csv", index=False)
    (tmp_path / "limits.json").write_text(json.dumps({"max_box_share": 0.5, "max_boxes_per_photo": 35,
                                                      "size_bins": [0.005, 0.02]}))
    common = ["--detections", str(tmp_path / "detections.csv"), "--split", "test", "--n-resamples", "5",
              "--reference", str(tmp_path / "reference.csv"), "--limits", str(tmp_path / "limits.json")]
    (tmp_path / "failed.json").write_text(json.dumps({"threshold": 0.5, "gate_passed": False}))
    with pytest.raises(SystemExit, match="recall gate"):
        evaluate_main([*common, "--selection", str(tmp_path / "failed.json"), "--output-dir", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()

    (tmp_path / "passed.json").write_text(json.dumps({"threshold": 0.5, "gate_passed": True}))
    assert evaluate_main([*common, "--selection", str(tmp_path / "passed.json"), "--output-dir", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / "detection_metrics.json").read_text())
    assert set(report["by_type"]) == {"cut wound", "scratch"} and "ap50" in report["overall"]["intervals"]
    finals = pd.read_csv(tmp_path / "out" / "final_boxes.csv")
    assert set(finals["relative_path"]) == set(photos.index)  # every photo gets a box, by fallback if needed
    results = pd.read_csv(tmp_path / "out" / "photo_results.csv")
    assert results["n_predicted"].sum() == len(finals)
