"""T28: macro precision and recall intervals on the T08 bootstrap's resamples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import Predictions, bootstrap_draws, compute_metrics
from src.evaluation.precision_recall import intervals


def predictions(seed: int = 0) -> Predictions:
    # 60 images of 20 cases over 4 classes; class 3 is never predicted
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 4, size=60)
    y_pred = np.where(rng.random(60) < 0.7, y_true, rng.integers(0, 3, size=60))
    y_pred[y_pred == 3] = 0
    probs = np.eye(4)[y_pred] * 0.7 + 0.075
    cases = np.repeat([f"c{i}" for i in range(20)], 3)
    return Predictions(frame=pd.DataFrame({"pseudonimized_id": cases}), classes=["a", "b", "c", "d"],
                       y_true=y_true, y_pred=y_pred, probs=probs, cases=cases)


def test_recall_interval_is_the_balanced_accuracy_interval():
    p = predictions()
    row = intervals(p, 200, 42)
    draws = bootstrap_draws(p, 200, 42)["balanced_accuracy"]
    assert row["macro_recall_ci_low"] == pytest.approx(np.percentile(draws, 2.5))
    assert row["macro_recall_ci_high"] == pytest.approx(np.percentile(draws, 97.5))
    # Observed values agree with compute_metrics; the unpredicted class counts 0 in precision
    metrics = compute_metrics(p)
    assert row["macro_precision"] == pytest.approx(metrics["macro_precision"])
    assert row["macro_recall"] == pytest.approx(metrics["balanced_accuracy"])
    assert row["macro_precision_ci_low"] <= row["macro_precision"] <= row["macro_precision_ci_high"]
