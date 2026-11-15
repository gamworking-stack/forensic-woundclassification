"""T35: pixel-based quality of the test crops against correct predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter

from src.evaluation.quality_accuracy import image_measures, lightness, logistic_slope, main
from src.evaluation.single_box import photo_stem
from tests.synthetic_data import build_synthetic_box_dataset, build_synthetic_dataset

CLASSES = ["contusion", "cut wound", "graze wound", "gunshot wound", "laceration", "penetration wound", "scratch"]


def test_lightness_and_sharpness():
    np.testing.assert_allclose(lightness(np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.uint8)), [[100, 0]], atol=0.01)
    assert lightness(np.array([[[119, 119, 119]]], dtype=np.uint8))[0, 0] == pytest.approx(50, abs=0.5)
    # Logistic slope: positive when higher x is more often correct; NaN when every prediction is right
    x = np.arange(10.0)
    assert logistic_slope(x, np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1.0])) > 0
    assert np.isnan(logistic_slope(x, np.ones(10)))


def test_blur_lowers_sharpness_not_brightness(tmp_path):
    rng = np.random.default_rng(0)
    sharp = Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8))
    sharp.save(tmp_path / "sharp.png")
    sharp.filter(ImageFilter.GaussianBlur(3)).save(tmp_path / "blur.png")
    a, b = image_measures(tmp_path / "sharp.png", 384), image_measures(tmp_path / "blur.png", 384)
    assert b["sharpness"] < a["sharpness"] / 5
    assert b["brightness"] == pytest.approx(a["brightness"], abs=2)
    assert (a["image_width"], a["image_height"]) == (300, 200)


@pytest.fixture()
def run(tmp_path):
    root = tmp_path / "wounds"
    data = build_synthetic_box_dataset(root, build_synthetic_dataset(root))
    manifest = pd.read_csv(data["box_manifest"], dtype=str)
    # Boxes as the real manifest stores them, matching each crop file's size
    sizes = [Image.open(data["box_root"] / p).size for p in manifest["roi_relative_path"]]
    manifest["x1"], manifest["y1"] = 10, 20
    manifest["x2"] = [10 + w for w, _ in sizes]
    manifest["y2"] = [20 + h for _, h in sizes]
    manifest["source_width"], manifest["source_height"] = 400, 300
    manifest.to_csv(data["box_manifest"], index=False)
    photos = manifest.drop_duplicates("relative_path")
    patients = tmp_path / "patient_cases.csv"
    pd.DataFrame({"relative_path": photos["relative_path"],
                  "patient_case": [str(i // 2) for i in range(len(photos))]}).to_csv(patients, index=False)

    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    area = (test["x2"].astype(int) - 10) * (test["y2"].astype(int) - 20)
    big = (area >= area.median()).to_numpy().copy()
    # Mostly right on the bigger crops, with one exception each way (perfect separation has no finite OR)
    first_big, first_small = np.flatnonzero(big)[0], np.flatnonzero(~big)[0]
    big[first_big], big[first_small] = False, True
    for family in ("vgg16", "resnet152"):
        # vgg16 is right on the bigger crops only; resnet152 on every crop
        right = big if family == "vgg16" else np.ones(len(test), bool)
        truth = test["canonical_wound_label"]
        wrong = [CLASSES[(CLASSES.index(t) + 1) % 7] for t in truth]
        frame = pd.DataFrame({"relative_path": test["roi_relative_path"], "pseudonimized_id": "1", "split": "test",
                              "true_class": truth, "predicted_class": np.where(right, truth, wrong)})
        for name in CLASSES:
            frame[f"prob_{name.replace(' ', '_')}"] = [0.7 if (p == name) else 0.05 for p in frame["predicted_class"]]
        (tmp_path / "preds" / family).mkdir(parents=True)
        frame.to_csv(tmp_path / "preds" / family / "test_predictions.csv", index=False)

        # Full-image run: one row per test photo, named by the raw folder ("raw/") and file, not the class path;
        # vgg16's full-image model is right on even-numbered groups only
        test_photos = test.drop_duplicates("relative_path")
        full_right = test_photos["pseudonimized_id"].astype(int) % 2 == 0 if family == "vgg16" else True
        full = pd.DataFrame({"relative_path": ["raw/" + photo_stem(f) + ".JPG" for f in test_photos["filename"]],
                             "pseudonimized_id": test_photos["pseudonimized_id"], "split": "test",
                             "true_class": test_photos["canonical_wound_label"]})
        other = np.where(full["true_class"] == "scratch", "contusion", "scratch")
        full["predicted_class"] = np.where(full_right, full["true_class"], other)
        (tmp_path / "full" / family).mkdir(parents=True)
        full.to_csv(tmp_path / "full" / family / "test_predictions.csv", index=False)
    return data, patients, tmp_path, big


def test_area_fraction_and_full_image_comparison(run):
    data, patients, tmp_path, big = run
    out = tmp_path / "out"
    assert main(["--predictions-root", str(tmp_path / "preds"), "--families", "vgg16", "resnet152",
                 "--box-manifest", str(data["box_manifest"]), "--patient-cases", str(patients),
                 "--full-predictions-root", str(tmp_path / "full"), "--measures", "area_fraction",
                 "--output-dir", str(out), "--bootstrap", "30"]) == 0
    crops = pd.read_csv(out / "quality_per_crop.csv", dtype={"crop": str})
    np.testing.assert_allclose(crops["area_fraction"], crops["crop_width"] * crops["crop_height"] / (400 * 300))
    manifest = pd.read_csv(data["box_manifest"], dtype=str).set_index("roi_relative_path")
    group = manifest.loc[crops["crop"], "pseudonimized_id"].astype(int).to_numpy()
    assert crops["vgg16_full_correct"].tolist() == (group % 2 == 0).tolist() and crops["resnet152_full_correct"].all()
    quint = pd.read_csv(out / "quality_quintiles.csv")
    assert set(quint["measure"]) == {"area_fraction"}
    assert np.allclose(quint["crop_minus_full"], quint["accuracy"] - quint["full_accuracy"])
    assert (out / "crop_vs_full_area_fraction.png").exists()
    assert "full_odds_ratio_per_sd" in pd.read_csv(out / "quality_correlation.csv").columns


def test_megapixels_without_images(run):
    data, patients, tmp_path, big = run
    out = tmp_path / "out"
    assert main(["--predictions-root", str(tmp_path / "preds"), "--families", "vgg16", "resnet152",
                 "--box-manifest", str(data["box_manifest"]), "--patient-cases", str(patients),
                 "--output-dir", str(out), "--bootstrap", "50"]) == 0
    crops = pd.read_csv(out / "quality_per_crop.csv")
    assert "sharpness" not in crops.columns and crops["vgg16_correct"].tolist() == big.tolist()
    corr = pd.read_csv(out / "quality_correlation.csv").query("measure == 'megapixels'").set_index("family")
    assert corr.loc["vgg16", "spearman_rho"] > 0.3 and corr.loc["vgg16", "odds_ratio_per_sd"] > 1.5
    assert np.isnan(corr.loc["resnet152", "odds_ratio_per_sd"])  # always right: no slope
    assert (out / "quality_scatter_megapixels.png").exists() and (out / "quality_summary.png").exists()


def test_image_measures_from_crops(run):
    data, patients, tmp_path, _ = run
    out = tmp_path / "out"
    assert main(["--predictions-root", str(tmp_path / "preds"), "--families", "vgg16", "resnet152",
                 "--box-manifest", str(data["box_manifest"]), "--patient-cases", str(patients),
                 "--image-root", str(data["box_root"]), "--output-dir", str(out), "--bootstrap", "20"]) == 0
    assert set(pd.read_csv(out / "quality_correlation.csv")["measure"]) == {"megapixels", "area_fraction", "sharpness", "brightness", "contrast"}
    quint = pd.read_csv(out / "quality_quintiles.csv")
    assert quint.groupby(["family", "measure"])["crops"].sum().eq(len(pd.read_csv(out / "quality_per_crop.csv"))).all()
    for measure in ("sharpness", "brightness", "contrast"):
        assert (out / f"quality_scatter_{measure}.png").exists()
