"""Tests for the evaluation module (hand-made prediction files, CPU)."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import SCENARIO_IDS, main as evaluate_main
from src.evaluation.metrics import (
    bootstrap_draws,
    case_bootstrap,
    class_report,
    compute_metrics,
    confusion_table,
    load_predictions,
    summary_metrics,
)


SEVEN = ["contusion", "cut wound", "graze wound", "gunshot wound", "laceration", "penetration wound", "scratch"]


def write_predictions(path: Path, split: str, classes, y_true, y_pred, cases=None, confidence=0.6):
    """Prediction CSV in the training format: the predicted class gets ``confidence``, the rest share."""
    n, k = len(y_true), len(classes)
    probs = np.full((n, k), (1.0 - confidence) / (k - 1))
    probs[np.arange(n), y_pred] = confidence
    frame = pd.DataFrame({
        "relative_path": [f"{split}/img_{i}.jpg" for i in range(n)],
        "pseudonimized_id": [str(c) for c in (cases if cases is not None else range(n))],
        "split": split,
        "true_class": [classes[i] for i in y_true],
        "predicted_class": [classes[i] for i in y_pred],
    })
    for j, name in enumerate(classes):
        frame[f"prob_{name.replace(' ', '_')}"] = probs[:, j]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_scenario(folder: Path, classes=SEVEN, per_class=3, case_offset=0, wrong=()):
    """Validation and test files; every image correct except the row indices in ``wrong``."""
    y_true = np.repeat(np.arange(len(classes)), per_class)
    y_pred = y_true.copy()
    for i in wrong:
        y_pred[i] = (y_true[i] + 1) % len(classes)
    cases = np.arange(len(y_true)) + case_offset
    for split in ("validation", "test"):
        write_predictions(folder / f"{split}_predictions.csv", split, classes, y_true, y_pred, cases)
    return folder


def test_perfect_predictions(tmp_path):
    folder = write_scenario(tmp_path / "s")
    p = load_predictions(folder / "test_predictions.csv", "test")
    metrics = compute_metrics(p)
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auc", "weighted_auc"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["n_images"] == 21 and metrics["n_cases"] == 21
    table = confusion_table(p).to_numpy()
    assert (table == np.diag(np.full(7, 3))).all()


def test_hand_computed_three_class_example(tmp_path):
    classes = ["a", "b", "c"]
    path = write_predictions(tmp_path / "test_predictions.csv", "test", classes,
                             y_true=[0, 0, 1, 1, 2, 2], y_pred=[0, 1, 1, 1, 2, 0])
    p = load_predictions(path, "test")
    assert confusion_table(p).to_numpy().tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]

    report = class_report(p).set_index("class")
    assert report.loc[["a", "b", "c"], "precision"].tolist() == pytest.approx([1 / 2, 2 / 3, 1.0])
    assert report.loc[["a", "b", "c"], "recall"].tolist() == pytest.approx([0.5, 1.0, 0.5])
    assert report.loc[["a", "b", "c"], "f1"].tolist() == pytest.approx([0.5, 0.8, 2 / 3])
    assert report.loc["macro avg", "f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)

    metrics = compute_metrics(p)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_bootstrap_is_reproducible(tmp_path):
    folder = write_scenario(tmp_path / "s", wrong=(0, 4, 9))
    p = load_predictions(folder / "test_predictions.csv", "test")
    first, second = case_bootstrap(p, 300, seed=7), case_bootstrap(p, 300, seed=7)
    assert first == second
    assert first["accuracy"]["low"] <= compute_metrics(p)["accuracy"] <= first["accuracy"]["high"]
    assert first["accuracy"]["n_resamples"] == 300


def test_bootstrap_keeps_images_of_a_case_together(tmp_path):
    # Case "A": 9 correct images; case "B": 1 wrong image. Whole cases can only give
    # accuracy 1.0 (A, A), 0.9 (A, B) or 0.0 (B, B).
    y_true = [0] * 5 + [1] * 5
    y_pred = [0] * 5 + [1] * 4 + [0]
    path = write_predictions(tmp_path / "test_predictions.csv", "test", ["x", "y"], y_true, y_pred,
                             cases=["A"] * 9 + ["B"])
    draws = bootstrap_draws(load_predictions(path, "test"), 200, seed=0)
    assert set(np.round(draws["accuracy"], 6)) <= {1.0, 0.9, 0.0}


def test_missing_class_in_resample_is_left_out_not_scored_zero():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
    metrics = summary_metrics(y, y, probs)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_auc"] == pytest.approx(1.0)


@pytest.mark.parametrize("problem, match", [
    ("sum", "sum to 1"),
    ("duplicate", "duplicate"),
    ("argmax", "argmax"),
    ("unknown", "no probability column"),
    ("split", "expected split"),
])
def test_input_checks(tmp_path, problem, match):
    path = write_predictions(tmp_path / "test_predictions.csv", "test", ["a", "b", "c"],
                             y_true=[0, 1, 2], y_pred=[0, 1, 2])
    frame = pd.read_csv(path)
    if problem == "sum":
        frame.loc[0, "prob_a"] = 0.9
    elif problem == "duplicate":
        frame.loc[1, "relative_path"] = frame.loc[0, "relative_path"]
    elif problem == "argmax":
        frame.loc[0, "predicted_class"] = "b"
    elif problem == "unknown":
        frame.loc[0, "true_class"] = "z"
    elif problem == "split":
        frame.loc[0, "split"] = "validation"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_predictions(path, "test")


def test_different_test_sets_are_an_error(tmp_path):
    root = tmp_path / "run"
    write_scenario(root / "full" / "resnet152")
    write_scenario(root / "roi" / "resnet152", case_offset=100)
    with pytest.raises(ValueError, match="Test sets differ"):
        evaluate_main(["--experiments-root", str(root), "--bootstrap", "50"])


def test_experiments_root_writes_index_and_skips_members(tmp_path):
    root = tmp_path / "run"
    write_scenario(root / "full" / "resnet152", wrong=(2,))
    write_scenario(root / "roi" / "resnet152")
    write_scenario(root / "roi" / "alexnet_ensemble", wrong=(5, 6))
    write_scenario(root / "roi" / "alexnet_ensemble" / "member_01")
    assert evaluate_main(["--experiments-root", str(root), "--bootstrap", "50"]) == 0

    index = pd.read_csv(root / "evaluation_index.csv")
    assert len(index) == len(SCENARIO_IDS) == 16
    evaluated = index[index["status"] == "evaluated"]
    assert sorted(evaluated["scenario_id"]) == ["S03", "S04", "S16"]
    assert (index["status"] == "missing").sum() == 13
    assert not index["folder"].fillna("").str.contains("member_01").any()

    folder = root / "roi" / "alexnet_ensemble"
    for split in ("validation", "test"):
        for name in ("metrics.json", "classification_report.csv", "confusion_matrix.csv", "confusion_matrix.png"):
            assert (folder / f"{split}_{name}").is_file()
    test_metrics = json.loads((folder / "test_metrics.json").read_text())
    assert test_metrics["bootstrap"]["unit"] == "case"
    assert set(test_metrics["confidence_intervals"]) == {"accuracy", "balanced_accuracy", "macro_f1", "macro_auc"}
    assert not (root / "roi" / "alexnet_ensemble" / "member_01" / "test_metrics.json").exists()
