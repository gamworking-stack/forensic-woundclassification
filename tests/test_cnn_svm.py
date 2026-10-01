"""Tests for the hybrid CNN-SVM entry point (synthetic images, CPU, no downloads)."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.training.cnn_svm import DEFAULT_EXTRACTORS, main as svm_main, svm_grid
from tests.synthetic_data import EXPECTED_PREDICTION_COLUMNS, IMAGES_PER_CLASS, RAW_FOLDERS, build_synthetic_dataset


def svm_args(data, output_dir, input_path="roi"):
    manifest = data["roi_manifest"] if input_path == "roi" else data["full_manifest"]
    root = data["roi_root"] if input_path == "roi" else data["raw_root"]
    return [
        "--input-path", input_path,
        "--manifest", str(manifest),
        "--image-root", str(root),
        "--roi-manifest", str(data["roi_manifest"]),
        "--output-dir", str(output_dir),
        "--image-size", "64",
        "--no-pretrained",
        "--device", "cpu",
        "--num-workers", "0",
        "--smoke",
    ]


@pytest.fixture(scope="module")
def svm_run(tmp_path_factory):
    data = build_synthetic_dataset(tmp_path_factory.mktemp("wounds"))
    out = tmp_path_factory.mktemp("svm")
    assert svm_main(svm_args(data, out)) == 0
    return data, out


def test_writes_all_outputs(svm_run):
    _, out = svm_run
    for name in ("run_config.json", "svm_validation_tuning.csv", "validation_predictions.csv",
                 "test_predictions.csv", "train.log"):
        assert (out / name).is_file(), f"missing {name}"
    for fam in DEFAULT_EXTRACTORS:
        for split in ("train", "validation", "test"):
            assert (out / "features" / f"{fam}_{split}.npz").is_file()

    tuning = pd.read_csv(out / "svm_validation_tuning.csv")
    assert len(tuning) == len(svm_grid()) == 20

    config = json.loads((out / "run_config.json").read_text())
    assert config["feature_dim"] == 4096 + 2048 + 1280
    assert config["best_params"] in svm_grid()


def test_prediction_files_match_training_format(svm_run):
    _, out = svm_run
    for split in ("validation", "test"):
        preds = pd.read_csv(out / f"{split}_predictions.csv")
        assert preds.columns.tolist() == EXPECTED_PREDICTION_COLUMNS
        assert len(preds) == IMAGES_PER_CLASS * len(RAW_FOLDERS)
        assert (preds["split"] == split).all()
        assert np.allclose(preds.filter(like="prob_").sum(axis=1), 1.0, atol=1e-4)


def test_test_features_extracted_only_after_svm_is_chosen(svm_run):
    _, out = svm_run
    log = (out / "train.log").read_text()
    chosen = log.index("Chosen SVM")
    assert "features for test" in log[chosen:]
    assert "features for test" not in log[:chosen]


def test_rerun_reuses_cached_features(svm_run, tmp_path):
    data, out = svm_run
    feature_files = sorted((out / "features").glob("*.npz"))
    mtimes = [f.stat().st_mtime_ns for f in feature_files]
    tuning_before = pd.read_csv(out / "svm_validation_tuning.csv")

    log_size = (out / "train.log").stat().st_size
    assert svm_main(svm_args(data, out)) == 0

    assert sorted((out / "features").glob("*.npz")) == feature_files
    assert [f.stat().st_mtime_ns for f in feature_files] == mtimes
    assert "Extracting" not in (out / "train.log").read_text()[log_size:]
    pd.testing.assert_frame_equal(pd.read_csv(out / "svm_validation_tuning.csv"), tuning_before)
