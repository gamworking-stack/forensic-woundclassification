"""T29: inconclusive predictions below a confidence threshold."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.confidence_threshold import (INCONCLUSIVE, VALUES, class_rows, confusion_rows, draws, main,
                                                 values)
from src.evaluation.metrics import Predictions, bootstrap_draws, compute_metrics


def predictions() -> Predictions:
    # 6 images of 3 cases, 2 classes; top probabilities 0.9, 0.55, 0.7, 0.58, 0.95, 0.65
    probs = np.array([[0.9, 0.1], [0.55, 0.45], [0.3, 0.7], [0.42, 0.58], [0.95, 0.05], [0.35, 0.65]])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    cases = np.repeat(["c1", "c2", "c3"], 2)
    return Predictions(frame=pd.DataFrame({"pseudonimized_id": cases}), classes=["a", "b"],
                       y_true=y_true, y_pred=probs.argmax(axis=1), probs=probs, cases=cases)


def test_inconclusive_counts_and_accuracy():
    p = predictions()
    conclusive = p.probs.max(axis=1) >= 0.6
    v = dict(zip(VALUES, values(p.y_true, p.y_pred, p.probs, conclusive)))
    # Images 2 and 4 are inconclusive; of the other 4, images 1 and 6 are right
    assert v["coverage"] == pytest.approx(4 / 6)
    assert v["accuracy"] == pytest.approx(2 / 4)
    assert v["accuracy_all"] == pytest.approx(2 / 6)
    rows = {r["true_class"]: r for r in class_rows({}, p, conclusive)}
    assert (rows["a"]["inconclusive"], rows["a"]["conclusive_correct"], rows["a"]["accuracy_no_threshold"]) == (1, 1, 2 / 3)
    table = confusion_rows({"family": "x"}, p, conclusive).set_index("true_class")
    assert table[INCONCLUSIVE].tolist() == [1, 1]
    assert table.loc["b", "a"] == 1 and table.loc["b", "b"] == 1


def test_threshold_zero_reproduces_the_saved_metrics():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=40)
    y_true = rng.integers(0, 3, size=40)
    cases = np.repeat([f"c{i}" for i in range(10)], 4)
    p = Predictions(frame=pd.DataFrame({"pseudonimized_id": cases}), classes=["a", "b", "c"],
                    y_true=y_true, y_pred=probs.argmax(axis=1), probs=probs, cases=cases)
    everything = np.ones(40, dtype=bool)
    v = dict(zip(VALUES, values(p.y_true, p.y_pred, p.probs, everything)))
    saved = compute_metrics(p)
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "macro_precision", "macro_recall"):
        assert v[name] == pytest.approx(saved[name])
    ours = draws(p, everything, 100, 42)
    theirs = bootstrap_draws(p, 100, 42)
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc"):
        np.testing.assert_allclose(ours[:, VALUES.index(name)], theirs[name])


def test_output_inside_the_repository_is_refused(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="inside the repository"):
        main(["--main-root", str(tmp_path), "--crop-root", str(tmp_path), "--reference-main", str(tmp_path),
              "--reference-crop", str(tmp_path), "--output-dir", str(root / "halloween")])
