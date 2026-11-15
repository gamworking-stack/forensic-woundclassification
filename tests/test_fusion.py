"""T36: soft voting between the full-image and crop models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.contrast_compare import main as compare_main
from src.evaluation.fusion import main as fusion_main
from src.evaluation.single_box import photo_stem
from tests.synthetic_data import build_synthetic_box_dataset, build_synthetic_dataset

CLASSES = ["contusion", "cut wound", "graze wound", "gunshot wound", "laceration", "penetration wound", "scratch"]
COLUMNS = [f"prob_{c.replace(' ', '_')}" for c in CLASSES]
FAMILIES = ["vgg16", "resnet152"]


def frame(paths, groups, truth, probs):
    out = pd.DataFrame({"relative_path": paths, "pseudonimized_id": groups, "split": "x", "true_class": truth,
                        "predicted_class": [CLASSES[k] for k in probs.argmax(axis=1)]})
    for k, c in enumerate(COLUMNS):
        out[c] = probs[:, k]
    return out


@pytest.fixture()
def runs(tmp_path):
    root = tmp_path / "wounds"
    data = build_synthetic_box_dataset(root, build_synthetic_dataset(root))
    boxes = pd.read_csv(data["box_manifest"], dtype=str)
    rng = np.random.default_rng(0)
    source = tmp_path / "runs"
    for family in FAMILIES:
        for split in ("validation", "test"):
            crops = boxes[boxes["split"] == split]
            photos = crops.drop_duplicates("relative_path")
            truth = photos["canonical_wound_label"].map(CLASSES.index).to_numpy()
            # The full-image model is right on every photo, but only just; the crop model is noise
            full_probs = np.full((len(photos), 7), 0.125)
            full_probs[np.arange(len(photos)), truth] = 0.25
            crop_probs = rng.dirichlet(np.ones(7), size=len(crops))
            full = frame(["raw/" + photo_stem(f) + ".JPG" for f in photos["filename"]], photos["pseudonimized_id"],
                         photos["canonical_wound_label"], full_probs)
            crop = frame(crops["roi_relative_path"], crops["pseudonimized_id"], crops["canonical_wound_label"], crop_probs)
            for run, path, table in (("main", "full", full), ("roi_crop", "crop", crop)):
                (source / run / path / family).mkdir(parents=True, exist_ok=True)
                table.assign(split=split).to_csv(source / run / path / family / f"{split}_predictions.csv", index=False)
    return data, source, tmp_path


def run_fusion(data, source, out):
    return fusion_main(["--source-root", str(source), "--output-root", str(out), "--box-manifest", str(data["box_manifest"]),
                        "--families", *FAMILIES])


def test_equal_weights_average_full_and_crops(runs):
    data, source, tmp_path = runs
    assert run_fusion(data, source, tmp_path / "fused") == 0
    full = pd.read_csv(source / "main/full/vgg16/test_predictions.csv", dtype={"pseudonimized_id": str})
    crop = pd.read_csv(source / "roi_crop/crop/vgg16/test_predictions.csv", dtype={"pseudonimized_id": str})
    photo_fused = pd.read_csv(tmp_path / "fused/equal/main/full/vgg16/test_predictions.csv", dtype={"pseudonimized_id": str})
    crop_fused = pd.read_csv(tmp_path / "fused/equal/roi_crop/crop/vgg16/test_predictions.csv", dtype={"pseudonimized_id": str})
    # Same rows, in the same order, as the files they replace
    assert photo_fused["relative_path"].tolist() == full["relative_path"].tolist()
    assert crop_fused["relative_path"].tolist() == crop["relative_path"].tolist()

    boxes = pd.read_csv(data["box_manifest"], dtype=str).set_index("roi_relative_path")
    stem = photo_stem(full["relative_path"][0])
    group = full["pseudonimized_id"][0]
    mine = [p for p in crop["relative_path"]
            if boxes.loc[p, "pseudonimized_id"] == group and photo_stem(boxes.loc[p, "filename"]) == stem]
    expected = 0.5 * full.loc[0, COLUMNS].to_numpy(float) + 0.5 * crop.set_index("relative_path").loc[mine, COLUMNS].to_numpy(float).mean(axis=0)
    np.testing.assert_allclose(photo_fused.loc[0, COLUMNS].to_numpy(float), expected)
    one = crop_fused.set_index("relative_path").loc[mine[0], COLUMNS].to_numpy(float)
    np.testing.assert_allclose(one, 0.5 * full.loc[0, COLUMNS].to_numpy(float) + 0.5 * crop.set_index("relative_path").loc[mine[0], COLUMNS].to_numpy(float))
    assert (photo_fused[COLUMNS].to_numpy().argmax(axis=1) == photo_fused["predicted_class"].map(CLASSES.index)).all()


def test_tuned_weight_follows_the_better_model_on_validation(runs):
    data, source, tmp_path = runs
    run_fusion(data, source, tmp_path / "fused")
    weights = pd.read_csv(tmp_path / "fused/fusion_weights.csv")
    tuned = weights[weights["tuned"]].set_index(["family", "level"])["weight_full"]
    # The full-image model is perfect and the crop model noise: the tuned weight leans to the full image
    assert (tuned > 0.5).all()
    best = weights.groupby(["family", "level"])["validation_macro_f1"].max()
    assert np.allclose(weights[weights["tuned"]].set_index(["family", "level"])["validation_macro_f1"], best)
    assert (tmp_path / "fused/tuned/roi_crop/crop/resnet152/validation_predictions.csv").exists()


def test_fused_files_score_with_contrast_compare(runs):
    data, source, tmp_path = runs
    run_fusion(data, source, tmp_path / "fused")
    assert compare_main(["--original-root", str(source), "--contrast-root", str(tmp_path / "fused/equal"),
                         "--output-dir", str(tmp_path / "compare"), "--bootstrap", "20", "--families", *FAMILIES,
                         "--name", "fused_equal"]) == 0
    table = pd.read_csv(tmp_path / "compare/fused_equal_vs_original.csv")
    assert set(table["input_path"]) == {"full", "crop"}
    # Noise crops pull the perfect full-image model down at the photo level and the fused crops up
    acc = table[(table["metric"] == "accuracy") & (table["family"] == "vgg16")].set_index("input_path")["difference"]
    assert acc["crop"] > 0


def test_output_inside_the_repo_and_missing_crops_are_refused(runs):
    data, source, tmp_path = runs
    with pytest.raises(ValueError, match="inside the repository"):
        run_fusion(data, source, Path(__file__).resolve().parents[1] / "fused_tmp")
    crop_file = source / "roi_crop/crop/vgg16/test_predictions.csv"
    crop = pd.read_csv(crop_file)
    first = crop["relative_path"][0]
    boxes = pd.read_csv(data["box_manifest"], dtype=str).set_index("roi_relative_path")
    photo = boxes.loc[first, "relative_path"]
    keep = [boxes.loc[p, "relative_path"] != photo for p in crop["relative_path"]]
    crop[keep].to_csv(crop_file, index=False)
    with pytest.raises(ValueError, match="photographs without crops"):
        run_fusion(data, source, tmp_path / "fused2")
