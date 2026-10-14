"""T17 per-box ROI experiment: photo weights, photo-level selection and summary (synthetic, CPU)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.evaluation.box_summary import main as box_summary_main
from src.evaluation.evaluate import FAMILY_ORDER, main as evaluate_main
from src.evaluation.metrics import load_predictions
from src.training.cnn_svm import main as svm_main
from src.training.engine import classification_metrics, load_checkpoint, weighted_loss
from src.training.ensemble import soft_vote
from src.training.photo import PHOTO_COLUMN, photo_mean, photo_prediction_frame, photo_weights
from src.training.train import compute_class_weights, main as train_main
from tests.synthetic_data import EXPECTED_PREDICTION_COLUMNS, build_synthetic_box_dataset, build_synthetic_dataset
from tests.test_evaluation import SEVEN, write_predictions
from tests.test_summary import build_run


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("wounds")
    base = build_synthetic_dataset(root)
    return {**base, **build_synthetic_box_dataset(root, base)}


def box_args(data, output_dir, *extra):
    return ["--input-path", "box", "--manifest", str(data["box_manifest"]), "--image-root", str(data["box_root"]),
            "--output-dir", str(output_dir), "--no-pretrained", "--device", "cpu", "--num-workers", "0",
            "--image-size", "64", *extra]


# ---------- photo helpers ----------

def test_photo_weights_sum_to_one_per_photo():
    photos = ["a", "b", "b", "c", "c", "c"]
    w = photo_weights(photos)
    assert w.tolist() == pytest.approx([1, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])
    assert pd.Series(w).groupby(photos).sum().tolist() == pytest.approx([1, 1, 1])


def test_photo_mean_averages_crops_and_rejects_mixed_labels():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    names, labels, mean = photo_mean(["p1", "p2", "p1"], np.array([0, 1, 0]), probs)
    assert names == ["p1", "p2"] and labels.tolist() == [0, 1]
    np.testing.assert_allclose(mean, [[0.65, 0.35], [0.2, 0.8]])
    with pytest.raises(ValueError, match="different labels"):
        photo_mean(["p1", "p1"], np.array([0, 1]), probs[:2])


def test_photo_prediction_frame_has_one_row_per_photo():
    crops = pd.DataFrame({
        "relative_path": ["c1", "c2", "c3"], PHOTO_COLUMN: ["p1", "p1", "p2"], "pseudonimized_id": ["1", "1", "2"],
        "split": "test", "true_class": ["cut wound", "cut wound", "scratch"],
        "predicted_class": ["cut wound", "scratch", "scratch"],
        "prob_cut_wound": [0.9, 0.2, 0.1], "prob_scratch": [0.1, 0.8, 0.9],
    })
    photos = photo_prediction_frame(crops)
    assert photos["relative_path"].tolist() == ["p1", "p2"]
    assert photos["n_crops"].tolist() == [2, 1]
    assert photos["prob_cut_wound"].tolist() == pytest.approx([0.55, 0.1])
    assert photos["predicted_class"].tolist() == ["cut wound", "scratch"]


def test_class_weights_count_photos_not_crops():
    labels = np.array([0, 0, 0, 1])  # class 0: one photo with 3 crops; class 1: one photo
    weights = photo_weights(["a", "a", "a", "b"])
    assert compute_class_weights(labels, 2, weights).tolist() == pytest.approx([1.0, 1.0])
    assert compute_class_weights(labels, 2).tolist() != pytest.approx([1.0, 1.0])


def test_weighted_loss_matches_criterion_with_equal_weights_and_duplicates():
    torch.manual_seed(0)
    logits, targets = torch.randn(5, 3), torch.tensor([0, 1, 2, 1, 0])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 0.5]), label_smoothing=0.05)
    assert weighted_loss(criterion, logits, targets, torch.ones(5)).item() == pytest.approx(
        criterion(logits, targets).item(), rel=1e-6)
    # weight 2 on a row equals that row appearing twice
    doubled = weighted_loss(criterion, logits, targets, torch.tensor([2.0, 1, 1, 1, 1]))
    repeated = criterion(torch.cat([logits, logits[:1]]), torch.cat([targets, targets[:1]]))
    assert doubled.item() == pytest.approx(repeated.item(), rel=1e-6)


# ---------- training ----------

def test_train_box_writes_crop_rows_and_selects_on_photos(data, tmp_path):
    assert train_main(["--family", "mobilenet_v3_small", *box_args(data, tmp_path, "--max-epochs", "2")]) == 0
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["selection_unit"] == "photo"
    manifest = pd.read_csv(data["box_manifest"], dtype=str)
    for split in ("validation", "test"):
        preds = pd.read_csv(tmp_path / f"{split}_predictions.csv", dtype=str)
        assert preds.columns.tolist() == EXPECTED_PREDICTION_COLUMNS[:1] + [PHOTO_COLUMN] + EXPECTED_PREDICTION_COLUMNS[1:]
        assert len(preds) == (manifest["split"] == split).sum()
        assert set(preds[PHOTO_COLUMN]) == set(manifest.loc[manifest["split"] == split, PHOTO_COLUMN])

    # best.pt's validation macro-F1 is the photo-level score of its validation predictions
    val = load_predictions(tmp_path / "validation_predictions.csv", "validation")
    photos = val.frame[PHOTO_COLUMN].to_numpy()
    _, y, probs = photo_mean(photos, val.y_true, val.probs)
    expected = classification_metrics(y, probs.argmax(axis=1), 7)["macro_f1"]
    best = load_checkpoint(tmp_path / "best.pt", torch.device("cpu"))
    assert best["validation_macro_f1"] == pytest.approx(expected, abs=1e-6)


def test_cnn_svm_box_runs_and_keeps_photo_column(data, tmp_path):
    assert svm_main([*box_args(data, tmp_path)]) == 0
    tuning = pd.read_csv(tmp_path / "svm_validation_tuning.csv")
    assert len(tuning) > 0
    preds = pd.read_csv(tmp_path / "test_predictions.csv", dtype=str)
    assert PHOTO_COLUMN in preds.columns
    assert len(preds) == (pd.read_csv(data["box_manifest"], dtype=str)["split"] == "test").sum()


def test_soft_vote_carries_photo_column():
    def frame(p):
        return pd.DataFrame({"relative_path": ["c1", "c2"], PHOTO_COLUMN: ["p1", "p1"], "pseudonimized_id": ["1", "1"],
                             "split": "test", "true_class": ["scratch", "scratch"], "predicted_class": ["scratch"] * 2,
                             "prob_cut_wound": [p, p], "prob_scratch": [1 - p, 1 - p]})
    voted = soft_vote([frame(0.2), frame(0.4)], ["m1", "m2"], ["cut wound", "scratch"])
    assert voted[PHOTO_COLUMN].tolist() == ["p1", "p1"]
    assert voted["prob_cut_wound"].tolist() == pytest.approx([0.3, 0.3])


# ---------- evaluation ----------

def write_box_scenario(folder: Path, crops_per_photo=2, wrong_photos=()):
    """Crop-level files for the 21 photos of tests.test_evaluation.write_scenario, ``crops_per_photo`` each."""
    y_photo = np.repeat(np.arange(7), 3)
    for split in ("validation", "test"):
        y_true = np.repeat(y_photo, crops_per_photo)
        y_pred = y_true.copy()
        for i in wrong_photos:
            y_pred[i * crops_per_photo:(i + 1) * crops_per_photo] = (y_photo[i] + 1) % 7
        cases = np.repeat(np.arange(21), crops_per_photo)
        path = write_predictions(folder / f"{split}_predictions.csv", split, SEVEN, y_true, y_pred, cases)
        frame = pd.read_csv(path, dtype=str)
        frame.insert(1, PHOTO_COLUMN, [f"{split}/photo_{c}.jpg" for c in cases])
        frame.to_csv(path, index=False)


def test_evaluate_names_box_scenarios_s17_to_s24(tmp_path):
    for family in FAMILY_ORDER:
        write_box_scenario(tmp_path / "box" / family)
    assert evaluate_main(["--experiments-root", str(tmp_path), "--bootstrap", "20"]) == 0
    index = pd.read_csv(tmp_path / "evaluation_index.csv")
    assert index["scenario_id"].tolist() == [f"S{i}" for i in range(17, 25)]
    assert (index["status"] == "evaluated").all() and (index["test_images"] == 42).all()


def test_box_summary_end_to_end(tmp_path):
    main = build_run(tmp_path / "main")
    box = tmp_path / "box_run"
    for family in FAMILY_ORDER:
        write_box_scenario(box / "box" / family, wrong_photos=(0, 5) if family == "cnn_svm" else ())
    assert evaluate_main(["--experiments-root", str(box), "--bootstrap", "20"]) == 0
    assert box_summary_main(["--box-root", str(box), "--main-root", str(main), "--bootstrap", "50"]) == 0

    out = box / "summary"
    for name in ("box_summary_table.csv", "box_vs_union.csv", "box_vs_full.csv", "box_vs_union_macro_f1.png"):
        assert (out / name).is_file(), name
    table = pd.read_csv(out / "box_summary_table.csv")
    assert table["scenario_id"].tolist() == [f"S{i}" for i in range(17, 25)]
    assert (table["test_photos"] == 21).all() and (table["test_crops"] == 42).all()
    versus = pd.read_csv(out / "box_vs_union.csv")
    f1 = versus[versus["metric"] == "macro_f1"].set_index("family")
    assert f1.loc["efficientnet_b0", "difference"] == pytest.approx(0.0)  # both perfect
    assert f1.loc["cnn_svm", "difference"] < 0  # two photos wrong in the box run
    photo = pd.read_csv(box / "box" / "cnn_svm" / "photo" / "test_predictions.csv")
    assert len(photo) == 21 and (photo["n_crops"] == 2).all()


def test_box_summary_rejects_other_test_cases(tmp_path):
    main = build_run(tmp_path / "main")
    box = tmp_path / "box_run"
    for family in FAMILY_ORDER:
        write_box_scenario(box / "box" / family, crops_per_photo=1)
    assert evaluate_main(["--experiments-root", str(box), "--bootstrap", "20"]) == 0
    # afterwards, give every box scenario a test photo from a case the main run never saw
    for family in FAMILY_ORDER:
        path = box / "box" / family / "test_predictions.csv"
        frame = pd.read_csv(path, dtype=str)
        frame.loc[0, "pseudonimized_id"] = "999"
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differs from the main run"):
        box_summary_main(["--box-root", str(box), "--main-root", str(main), "--bootstrap", "20"])
