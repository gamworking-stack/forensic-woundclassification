"""Smoke, resume and integrity tests for the shared training library (synthetic images, CPU)."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.training.config import FAMILIES
from src.training.train import main as train_main
from src.training.transforms import PadToSquare
from tests.synthetic_data import (
    EXPECTED_PREDICTION_COLUMNS,
    IMAGES_PER_CLASS,
    RAW_FOLDERS,
    build_synthetic_dataset,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return build_synthetic_dataset(tmp_path_factory.mktemp("wounds"))


def run_args(family, input_path, data, output_dir, *extra):
    # Pick the manifest and image folder for the full or ROI input
    manifest = data["roi_manifest"] if input_path == "roi" else data["full_manifest"]
    root = data["roi_root"] if input_path == "roi" else data["raw_root"]
    # CPU-only, untrained, single-process command line pointing at the synthetic data
    args = [
        "--family", family,
        "--input-path", input_path,
        "--manifest", str(manifest),
        "--image-root", str(root),
        "--roi-manifest", str(data["roi_manifest"]),
        "--output-dir", str(output_dir),
        "--no-pretrained",
        "--device", "cpu",
        "--num-workers", "0",
    ]
    # Small images keep CNN tests fast; ViT keeps its fixed 224 px input
    if family != "vit_b16":
        args += ["--image-size", "64"]
    return args + list(extra)


@pytest.mark.parametrize("family", FAMILIES)
def test_smoke_every_family(family, dataset, tmp_path):
    # A two-epoch smoke run must finish for this family
    assert train_main(run_args(family, "roi", dataset, tmp_path, "--smoke")) == 0

    # Every output file of a run is written
    for name in ("run_config.json", "training_history.csv", "validation_predictions.csv",
                 "test_predictions.csv", "best.pt", "last.pt", "train.log"):
        assert (tmp_path / name).is_file(), f"missing {name}"

    # One history row per epoch
    history = pd.read_csv(tmp_path / "training_history.csv")
    assert history["epoch"].tolist() == [1, 2]

    # Prediction files: fixed columns, one row per image, right split, probabilities summing to 1
    for split in ("validation", "test"):
        preds = pd.read_csv(tmp_path / f"{split}_predictions.csv")
        assert preds.columns.tolist() == EXPECTED_PREDICTION_COLUMNS
        assert len(preds) == IMAGES_PER_CLASS * len(RAW_FOLDERS)
        assert (preds["split"] == split).all()
        prob_sums = preds.filter(like="prob_").sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-4)


def test_full_scope_matched_drops_images_without_roi(dataset, tmp_path):
    # Train on full images with each scope for one epoch
    for scope, train_images in (("matched", 14), ("all", 21)):
        out = tmp_path / scope
        args = run_args("mobilenet_v3_small", "full", dataset, out, "--max-epochs", "1", "--full-scope", scope)
        assert train_main(args) == 0
        # Matched scope keeps only images that also have an ROI crop; validation and test are always matched
        config = json.loads((out / "run_config.json").read_text())
        assert config["images_per_split"] == {"train": train_images, "validation": 14, "test": 14}
        assert config["classes"] == sorted(RAW_FOLDERS)


def test_resume_continues_without_duplicating_history(dataset, tmp_path):
    # Train one epoch, then resume up to two
    base = run_args("mobilenet_v3_small", "roi", dataset, tmp_path)
    assert train_main(base + ["--max-epochs", "1"]) == 0
    assert train_main(base + ["--max-epochs", "2", "--resume"]) == 0

    # History has each epoch once, time keeps growing, and the log shows the resume
    history = pd.read_csv(tmp_path / "training_history.csv")
    assert history["epoch"].tolist() == [1, 2]
    assert history["elapsed_minutes"].is_monotonic_increasing
    assert "Resuming from epoch 2" in (tmp_path / "train.log").read_text()

    # Resuming a finished run does nothing and adds no rows
    assert train_main(base + ["--max-epochs", "2", "--resume"]) == 0
    assert "nothing to do" in (tmp_path / "train.log").read_text()
    assert len(pd.read_csv(tmp_path / "training_history.csv")) == 2


def test_missing_image_is_an_error(tmp_path):
    # Delete one ROI crop listed in the manifest
    data = build_synthetic_dataset(tmp_path / "data")
    roi = pd.read_csv(data["roi_manifest"], dtype=str)
    missing = roi.loc[0, "roi_relative_path"]
    (data["roi_root"] / missing).unlink()
    # Training must stop and name the missing file
    with pytest.raises(FileNotFoundError, match=missing):
        train_main(run_args("alexnet", "roi", data, tmp_path / "out", "--smoke"))


def test_case_leakage_is_an_error(tmp_path):
    # Give a train image the case ID of a test image
    data = build_synthetic_dataset(tmp_path / "data")
    roi = pd.read_csv(data["roi_manifest"], dtype=str)
    test_case = roi.loc[roi["split"] == "test", "pseudonimized_id"].iloc[0]
    roi.loc[roi.index[0], "pseudonimized_id"] = test_case  # row 0 is a train image
    roi.to_csv(data["roi_manifest"], index=False)
    # Training must refuse a manifest where one case spans two splits
    with pytest.raises(AssertionError, match="leakage"):
        train_main(run_args("alexnet", "roi", data, tmp_path / "out", "--smoke"))


def test_pad_to_square_centres_image():
    # A wide red image padded to a square
    image = Image.new("RGB", (40, 20), (255, 0, 0))
    padded = PadToSquare()(image)
    # Image stays centred: red in the middle band, black bars above and below
    assert padded.size == (40, 40)
    assert padded.getpixel((20, 20)) == (255, 0, 0)
    assert padded.getpixel((20, 5)) == (0, 0, 0)
    assert padded.getpixel((20, 34)) == (0, 0, 0)
    assert padded.getpixel((20, 10)) == (255, 0, 0)
    assert padded.getpixel((20, 29)) == (255, 0, 0)
