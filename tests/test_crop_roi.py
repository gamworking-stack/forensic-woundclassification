"""T18 per-crop ROI experiment: the per-box crops, each trained, selected and scored on its own."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.crop_summary import main as crop_summary_main
from src.evaluation.evaluate import FAMILY_ORDER, main as evaluate_main
from src.evaluation.metrics import load_predictions
from src.training.cnn_svm import main as svm_main
from src.training.engine import classification_metrics, load_checkpoint
from src.training.photo import PHOTO_COLUMN
from src.training.train import compute_class_weights, main as train_main
from tests.synthetic_data import EXPECTED_PREDICTION_COLUMNS, build_synthetic_box_dataset, build_synthetic_dataset
from tests.test_verify import BOOT, clean, write_ensemble, write_svm, write_tuned  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    # Synthetic photos plus their per-box crops, built once for the whole module
    root = tmp_path_factory.mktemp("wounds")
    base = build_synthetic_dataset(root)
    return {**base, **build_synthetic_box_dataset(root, base)}


def crop_args(data, output_dir, *extra):
    # Crop input on the box manifest, small CPU run with no pretrained weights; extra flags appended
    return ["--input-path", "crop", "--manifest", str(data["box_manifest"]), "--image-root", str(data["box_root"]),
            "--output-dir", str(output_dir), "--no-pretrained", "--device", "cpu", "--num-workers", "0",
            "--image-size", "64", *extra]


# ---------- training ----------

def test_train_crop_scores_each_crop_without_photo_weights(data, tmp_path):
    # Train a small CNN on crops for two epochs
    assert train_main(["--family", "mobilenet_v3_small", *crop_args(data, tmp_path, "--max-epochs", "2")]) == 0
    # Crop mode selects on crops with no photo weights in the loss, and keeps no photo counts
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["selection_unit"] == "crop" and config["loss_weights"] == "none"
    assert "photos_per_split" not in config

    # Each prediction file has exactly one row per crop of its split
    manifest = pd.read_csv(data["box_manifest"], dtype=str)
    for split in ("validation", "test"):
        preds = pd.read_csv(tmp_path / f"{split}_predictions.csv", dtype=str)
        assert preds.columns.tolist() == EXPECTED_PREDICTION_COLUMNS  # no photo column: scored per crop
        rows = manifest[manifest["split"] == split]
        assert sorted(preds["relative_path"]) == sorted(rows["roi_relative_path"])

    # class weights come from unweighted crop counts
    classes = config["classes"]
    labels = manifest.loc[manifest["split"] == "train", "canonical_wound_label"].map(classes.index).to_numpy()
    assert config["class_weights"] == pytest.approx(compute_class_weights(labels, len(classes)).tolist(), abs=1e-6)

    # best.pt's validation macro-F1 is the crop-level score of its validation predictions
    val = load_predictions(tmp_path / "validation_predictions.csv", "validation")
    expected = classification_metrics(val.y_true, val.probs.argmax(axis=1), 7)["macro_f1"]
    best = load_checkpoint(tmp_path / "best.pt", torch.device("cpu"))
    assert best["validation_macro_f1"] == pytest.approx(expected, abs=1e-6)


def test_cnn_svm_crop_writes_crop_rows(data, tmp_path):
    assert svm_main([*crop_args(data, tmp_path)]) == 0
    # CNN-SVM in crop mode also writes one row per test crop with no photo column
    preds = pd.read_csv(tmp_path / "test_predictions.csv", dtype=str)
    assert PHOTO_COLUMN not in preds.columns
    assert len(preds) == (pd.read_csv(data["box_manifest"], dtype=str)["split"] == "test").sum()


# ---------- summary and verification on a synthetic run ----------

def crop_manifest(root: Path) -> Path:
    """A manifest whose crops are the prediction files' rows (validation and test)."""
    # Collect every crop path from the EfficientNet-B0 validation and test predictions
    rows = []
    for split in ("validation", "test"):
        frame = pd.read_csv(root / "crop" / "efficientnet_b0" / f"{split}_predictions.csv", dtype=str)
        rows += [{"roi_relative_path": r, "split": split} for r in frame["relative_path"]]
    # Save them as the crop manifest the verifier checks against
    path = root / "crop_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def crop_run(clean, tmp_path_factory):
    # Copy the clean synthetic main run so this module can add a roi_crop tree without touching it
    base, _ = clean
    work = tmp_path_factory.mktemp("t18")
    shutil.copytree(base, work / "run", copy_function=shutil.copy2)
    base = work / "run"
    # Write fake finished crop runs for every family (CNN-SVM and ensemble have their own writers)
    root = base / "roi_crop"
    for family in FAMILY_ORDER:
        folder = root / "crop" / family
        folder.mkdir(parents=True)
        {"cnn_svm": write_svm, "alexnet_ensemble": write_ensemble}.get(family, write_tuned)(folder, (2,))
    # Mark the single-CNN runs as crop-selected with no loss weights, as real crop runs would be
    for path in root.rglob("run_config.json"):
        config = json.loads(path.read_text())
        if config.get("family") not in ("cnn_svm", "ensemble"):
            path.write_text(json.dumps({**config, "selection_unit": "crop", "loss_weights": "none"}))
    # Evaluate and summarise the crop experiment, then hand back the run and its crop manifest
    assert evaluate_main(["--experiments-root", str(root), "--bootstrap", str(BOOT)]) == 0
    assert crop_summary_main(["--crop-root", str(root)]) == 0
    return base, crop_manifest(root)


def run_verify_crop(base: Path, manifest: Path) -> pd.DataFrame:
    from src.evaluation.verify_crop import main as verify_crop_main
    # Run the crop verifier on a run folder
    code = verify_crop_main(["--crop-root", str(base / "roi_crop"), "--main-root", str(base / "main"),
                             "--split-assignments", str(base / "data" / "assignments.csv"),
                             "--crop-manifest", str(manifest), "--bootstrap", str(BOOT)])
    # Its exit code must be 1 exactly when the report has a FAIL row
    report = pd.read_csv(base / "roi_crop" / "verification" / "verification_report.csv", keep_default_na=False)
    assert code == (1 if (report["status"] == "FAIL").any() else 0)
    return report


def test_evaluate_names_crop_scenarios_s25_s32(crop_run):
    base, _ = crop_run
    # Crop scenarios are numbered S25 to S32, all with crop input and all evaluated
    index = pd.read_csv(base / "roi_crop" / "evaluation_index.csv")
    assert index["scenario_id"].tolist() == [f"S{25 + i}" for i in range(8)]
    assert (index["input_path"] == "crop").all() and (index["status"] == "evaluated").all()
    # Summary table follows family order, matches the evaluated F1 and carries crop counts and CIs
    table = pd.read_csv(base / "roi_crop" / "summary" / "crop_summary_table.csv")
    assert table["family"].tolist() == list(FAMILY_ORDER)
    assert np.allclose(table["macro_f1"], index["test_macro_f1"])
    assert {"test_crops", "macro_f1_ci_low", "macro_f1_ci_high"} <= set(table.columns)


def test_verify_crop_clean_run_passes(crop_run):
    report = run_verify_crop(*crop_run)
    # A clean run has no failures, and every check group was run
    bad = report[report["status"] == "FAIL"]
    assert bad.empty, bad.to_string()
    assert set(report["check"]) == {"files", "test set", "selection", "budget", "numbers"}


def test_verify_crop_catches_an_edited_number(crop_run, tmp_path):
    base, manifest = crop_run
    # Copy the run and nudge one macro F1 in the summary table
    shutil.copytree(base, tmp_path / "run", copy_function=shutil.copy2)
    path = tmp_path / "run" / "roi_crop" / "summary" / "crop_summary_table.csv"
    table = pd.read_csv(path)
    table.loc[0, "macro_f1"] += 0.01
    table.to_csv(path, index=False)
    # The verifier must flag exactly that table
    report = run_verify_crop(tmp_path / "run", manifest)
    assert report.loc[report["status"] == "FAIL", "item"].tolist() == ["crop_summary_table.csv"]


def test_verify_crop_rejects_photo_level_files(crop_run, tmp_path):
    base, manifest = crop_run
    # Copy the run and add a photo column to VGG16's test file, as if it were scored per photo
    shutil.copytree(base, tmp_path / "run", copy_function=shutil.copy2)
    path = tmp_path / "run" / "roi_crop" / "crop" / "vgg16" / "test_predictions.csv"
    frame = pd.read_csv(path, dtype=str)
    frame.insert(1, PHOTO_COLUMN, "photo.jpg")
    frame.to_csv(path, index=False)
    # The verifier must reject that scenario's test crops
    report = run_verify_crop(tmp_path / "run", manifest)
    assert "S29 test crops" in report.loc[report["status"] == "FAIL", "item"].tolist()
