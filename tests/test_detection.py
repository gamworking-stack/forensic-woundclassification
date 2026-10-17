"""T19 box rules: manual reference boxes from CVAT exports, and the helpers every detector arm uses."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.create_roi_dataset import build_roi_records, compute_union_box
from src.detection.boxes import (
    as_boxes,
    candidate_boxes,
    final_boxes,
    greedy_match,
    iou_matrix,
    main as boxes_main,
    nms,
    padded_crop_box,
    read_manual_boxes,
    training_limits,
)
from tests.test_roi_preprocessing import manifest_row, write_yolo_zip


# ---------- manual reference boxes ----------

PHOTOS = [  # case, filename, raw label, label, split, YOLO boxes, (width, height)
    ("1", "IMG_0001", "cut", "cut wound", "train", ["0 0.30 0.40 0.20 0.10", "0 0.70 0.60 0.10 0.30"], (4000, 3000)),
    ("2", "IMG_0002", "cut", "cut wound", "train", ["0 0.50 0.50 0.60 0.50"], (3000, 4000)),
    ("3", "IMG_0003", "cut", "cut wound", "test", ["0 0.10 0.10 0.05 0.05"], (6000, 4000)),
]


@pytest.fixture()
def reference_inputs(tmp_path):
    """A CVAT export with one unmatched photo and a per-box manifest built from it by T03's code."""
    # YOLO labels per photo, plus one photo the manifest does not list
    lines = {filename: boxes for _, filename, _, _, _, boxes, _ in PHOTOS}
    lines["NOT_IN_MANIFEST"] = ["0 0.5 0.5 0.1 0.1"]
    export = write_yolo_zip(tmp_path / "task_1.zip", ["cut wound"], lines)
    # Build the per-box crop manifest with T03's own code, so the reference must agree with real crops
    source = [manifest_row(case, filename, raw, label, split) for case, filename, raw, label, split, _, _ in PHOTOS]
    dimensions = {(label, filename): size for _, filename, _, label, _, _, size in PHOTOS}
    result = build_roi_records(source, [export], dimensions=dimensions, crop_mode="individual")
    manifest = tmp_path / "roi_manifest.csv"
    pd.DataFrame(result.rows).to_csv(manifest, index=False)
    return export, manifest


def test_reference_boxes_are_unpadded_and_reproduce_every_crop(reference_inputs):
    # Read the manual boxes: 4 boxes over 3 photos, the unlisted photo skipped
    export, manifest = reference_inputs
    reference = read_manual_boxes([export], manifest)
    assert len(reference) == 4 and reference["relative_path"].nunique() == 3  # unmatched photo skipped
    # The first box is the raw YOLO box in source pixels, without the crop padding
    first = reference[(reference["relative_path"] == "cut wound/IMG_0001") & (reference["box_index"] == 1)].iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == pytest.approx((800, 1050, 1600, 1350))
    # Padding each reference box the T03 way must give back its manifest crop exactly
    crops = pd.read_csv(manifest)
    for row in reference.to_dict("records"):
        crop = crops[(crops["relative_path"] == row["relative_path"]) & (crops["crop_index"] == row["box_index"])].iloc[0]
        assert padded_crop_box([[row["x1"], row["y1"], row["x2"], row["y2"]]], row["source_width"],
                               row["source_height"]) == (crop.x1, crop.y1, crop.x2, crop.y2)


def test_reference_boxes_refuse_a_manifest_that_disagrees(reference_inputs, tmp_path):
    export, manifest = reference_inputs
    # Shift one crop by a pixel: the reference must refuse a manifest that disagrees with CVAT
    edited = pd.read_csv(manifest, dtype=str)
    edited.loc[0, "x1"] = str(int(edited.loc[0, "x1"]) + 1)
    edited.to_csv(tmp_path / "edited.csv", index=False)
    with pytest.raises(ValueError, match="padded CVAT box"):
        read_manual_boxes([export], tmp_path / "edited.csv")
    # Drop one crop of a two-box photo: box counts no longer match, which must also be refused
    fewer = pd.read_csv(manifest, dtype=str)
    fewer = fewer[~((fewer["filename"] == "IMG_0001") & (fewer["crop_index"] == "2"))]
    fewer.to_csv(tmp_path / "fewer.csv", index=False)
    with pytest.raises(ValueError, match="2 CVAT boxes but 1 manifest crops"):
        read_manual_boxes([export], tmp_path / "fewer.csv")


def test_reference_boxes_need_every_manifest_photo_annotated(reference_inputs, tmp_path):
    export, manifest = reference_inputs
    # Add a manifest photo that has no CVAT annotation; it must be an error, not silently skipped
    extra = pd.read_csv(manifest, dtype=str)
    ghost = extra.iloc[0].copy()
    ghost["filename"], ghost["relative_path"] = "IMG_9999", "cut wound/IMG_9999"
    pd.concat([extra, ghost.to_frame().T]).to_csv(tmp_path / "ghost.csv", index=False)
    with pytest.raises(ValueError, match="no annotation"):
        read_manual_boxes([export], tmp_path / "ghost.csv")


def test_training_limits_and_script_output(reference_inputs, tmp_path):
    export, manifest = reference_inputs
    # Limits come from the training photos only: box cap, largest box share and the size tertiles
    limits = training_limits(read_manual_boxes([export], manifest))
    assert limits["max_boxes_per_photo"] == 2 and limits["train_photos"] == 2 and limits["train_boxes"] == 3
    assert limits["max_box_share"] == pytest.approx(0.30)  # 0.6 x 0.5 on IMG_0002; the test photo is ignored
    assert limits["size_bins"] == pytest.approx(np.quantile([0.20 * 0.10, 0.10 * 0.30, 0.60 * 0.50], [1 / 3, 2 / 3]))
    # The script writes the same limits plus box counts per split, and all 4 reference boxes
    out = tmp_path / "out"
    assert boxes_main(["--exports", str(export), "--box-manifest", str(manifest), "--output-dir", str(out)]) == 0
    written = json.loads((out / "reference_limits.json").read_text())
    assert written["max_boxes_per_photo"] == 2 and written["boxes_by_split"] == {"test": 1, "train": 3}
    assert len(pd.read_csv(out / "reference_boxes.csv")) == 4


# ---------- box helpers ----------

def test_padded_crop_box_matches_t03_for_pixel_boxes():
    # Random YOLO boxes on random photo sizes: padding pixel boxes must equal T03's union-box crop
    rng = np.random.default_rng(0)
    for _ in range(200):
        width, height = int(rng.integers(100, 6000)), int(rng.integers(100, 6000))
        yolo = [[*rng.uniform(0.05, 0.95, 2), *rng.uniform(0.01, 0.3, 2)] for _ in range(int(rng.integers(1, 4)))]
        pixels = [[(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]
                  for cx, cy, w, h in yolo]
        assert padded_crop_box(pixels, width, height) == compute_union_box(yolo, width, height, 0.08)


def test_iou_matrix():
    # Identical, half-overlapping and disjoint boxes; a zero-area box has IoU 0 with everything
    a = [[0, 0, 10, 10], [0, 0, 0, 0]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]
    np.testing.assert_allclose(iou_matrix(a, b), [[1, 50 / 150, 0], [0, 0, 0]])
    # No boxes gives an empty matrix, and a box with x2 < x1 is rejected
    assert iou_matrix([], b).shape == (0, 3)
    with pytest.raises(ValueError):
        as_boxes([10, 0, 0, 10])


def test_nms_keeps_the_best_of_each_cluster_and_breaks_ties_by_order():
    # Overlapping boxes collapse to the highest score; equal scores keep the earlier box
    boxes = [[0, 0, 10, 10], [1, 0, 11, 10], [20, 20, 30, 30], [0, 0, 10, 10]]
    assert nms(boxes, [0.9, 0.95, 0.5, 0.95]).tolist() == [1, 2]  # 1 before 3 at equal score; 0 suppressed by 1
    # The IoU threshold decides whether a half-overlapping pair is suppressed
    assert nms([[0, 0, 10, 10], [5, 0, 15, 10]], [0.9, 0.8], iou_threshold=0.3).tolist() == [0]  # IoU 1/3 > 0.3
    assert nms([[0, 0, 10, 10], [5, 0, 15, 10]], [0.9, 0.8], iou_threshold=0.5).tolist() == [0, 1]
    assert nms([], []).tolist() == []


def test_candidate_boxes_drop_oversize_then_nms_then_top_k():
    # Whole-image box, an overlapping pair and two separate boxes, scored high to low
    boxes = [[0, 0, 100, 100], [0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60], [70, 70, 80, 80]]
    scores = [0.99, 0.8, 0.7, 0.6, 0.5]
    # Order of rules: drop oversize, then NMS, then keep the top 2
    kept, kept_scores = candidate_boxes(boxes, scores, 100, 100, max_share=0.5, top_k=2)
    np.testing.assert_allclose(kept, [[0, 0, 10, 10], [50, 50, 60, 60]])  # whole-image box dropped, overlap suppressed
    np.testing.assert_allclose(kept_scores, [0.8, 0.6])
    # Box and score counts must agree
    with pytest.raises(ValueError, match="scores"):
        candidate_boxes(boxes, scores[:2], 100, 100, max_share=0.5)


def test_final_boxes_threshold_cap_and_fallbacks():
    # The cap of 1 keeps only the best box above the threshold
    boxes, scores = [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], [0.2, 0.9, 0.5]
    kept, kept_scores, fallback = final_boxes(boxes, scores, 10, 10, threshold=0.5, max_boxes=1)
    assert fallback == "none" and kept_scores.tolist() == [0.9] and kept.tolist() == [[2, 2, 3, 3]]
    kept, kept_scores, fallback = final_boxes(boxes, scores, 10, 10, threshold=0.5, max_boxes=5)
    assert kept_scores.tolist() == [0.9, 0.5]  # the threshold itself is kept
    # Nothing above the threshold: fall back to the single top box
    kept, kept_scores, fallback = final_boxes(boxes, scores, 10, 10, threshold=0.95, max_boxes=5)
    assert fallback == "top_box" and kept_scores.tolist() == [0.9]
    # No boxes at all: fall back to the full image, with no score
    kept, kept_scores, fallback = final_boxes([], [], 640, 480, threshold=0.5, max_boxes=5)
    assert fallback == "full_image" and kept.tolist() == [[0, 0, 640, 480]] and np.isnan(kept_scores[0])


def test_greedy_match_is_one_to_one_by_score():
    # Two manual boxes; predictions include a near-duplicate pair and one far away
    truth = [[0, 0, 10, 10], [20, 0, 30, 10]]
    preds = [[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10], [50, 50, 60, 60]]
    # Higher-scoring predictions claim truths first and each truth is matched at most once
    matches = greedy_match(preds, [0.5, 0.9, 0.8, 0.7], truth)
    assert [(p, t) for p, t, _ in matches] == [(1, 0), (2, 1)]  # pred 0 overlaps truth 0 too but comes later
    # Below the IoU threshold there is no match; exact overlap still matches at threshold 1.0
    assert greedy_match([[5, 0, 15, 10]], [1.0], [[0, 0, 10, 10]]) == []  # IoU 1/3 < 0.5
    assert greedy_match([[0, 0, 10, 10]], [1.0], [[0, 0, 10, 10]], iou_threshold=1.0)[0][2] == pytest.approx(1.0)
    assert greedy_match([], [], truth) == []
