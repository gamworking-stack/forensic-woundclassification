"""T15 summary table and paired full vs ROI comparison (hand-made prediction files, CPU)."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import FAMILY_ORDER, main as evaluate_main
from src.evaluation.summarize import POOLED, holm, load_pairs, main as summarize_main
from tests.test_evaluation import write_predictions, write_scenario, SEVEN


EVERY_OTHER = tuple(range(0, 21, 2))  # 11 of 21 test images wrong


def build_run(root: Path, full_wrong=(), roi_wrong=()) -> Path:
    """All 16 scenarios with the same 21 test cases, then T08 on them."""
    # Same case IDs on both paths for every family, each with its own list of mistakes
    for family in FAMILY_ORDER:
        write_scenario(root / "full" / family, wrong=full_wrong)
        write_scenario(root / "roi" / family, wrong=roi_wrong)
    # Run the T08 evaluation so the summary has metrics files to read
    assert evaluate_main(["--experiments-root", str(root), "--bootstrap", "50"]) == 0
    return root


def summarize(root: Path) -> pd.DataFrame:
    # Run the T15 summary and return its full-vs-ROI comparison table
    assert summarize_main(["--experiments-root", str(root), "--bootstrap", "200"]) == 0
    return pd.read_csv(root / "summary" / "full_vs_roi.csv")


def test_identical_paths_give_zero_difference(tmp_path):
    # Both paths make the same mistakes, so nothing should differ
    comparison = summarize(build_run(tmp_path, full_wrong=(0, 5), roi_wrong=(0, 5)))
    # Three metrics for each family plus the pooled row
    assert len(comparison) == 3 * (len(FAMILY_ORDER) + 1)
    # Zero difference with a zero-width interval and p = 1
    assert np.allclose(comparison[["difference", "ci_low", "ci_high"]], 0.0)
    assert (comparison["p_value"] == 1.0).all()


def test_better_roi_gives_positive_difference(tmp_path):
    # Full path gets 11 of 21 wrong, ROI path gets all right
    comparison = summarize(build_run(tmp_path, full_wrong=EVERY_OTHER))
    f1 = comparison[comparison["metric"] == "macro_f1"].set_index("family")
    # ROI wins clearly for every family: positive difference, interval above 0, p < 0.05
    assert (f1["roi"] == 1.0).all()
    assert (f1["difference"] > 0).all() and (f1["ci_low"] > 0).all()
    assert (f1["p_value"] < 0.05).all()
    # Holm only adjusts the per-family tests and can only raise p-values
    assert f1.loc[POOLED, "p_holm"] != f1.loc[POOLED, "p_holm"]  # no Holm for the pooled row
    assert (f1.drop(POOLED)["p_holm"] >= f1.drop(POOLED)["p_value"]).all()
    # The comparison figure is drawn
    assert (tmp_path / "summary" / "full_vs_roi_macro_f1.png").is_file()


def test_summary_table_copies_t08_metrics(tmp_path):
    # Build and summarise a run, then read the summary table
    root = build_run(tmp_path, full_wrong=EVERY_OTHER)
    summarize(root)
    table = pd.read_csv(root / "summary" / "summary_table.csv")
    # All 16 scenarios in order
    assert table["scenario_id"].tolist() == [f"S{i:02d}" for i in range(1, 17)]
    # Each row copies the scenario's T08 numbers rather than recomputing them
    for row in table.to_dict("records"):
        metrics = json.loads((root / row["input_path"] / row["family"] / "test_metrics.json").read_text())
        assert row["macro_f1"] == pytest.approx(metrics["macro_f1"])
        assert row["macro_f1_ci_low"] == pytest.approx(metrics["confidence_intervals"]["macro_f1"]["low"])
        assert row["test_cases"] == 21


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_different_test_cases_are_an_error(tmp_path):
    # Overwrite one ROI scenario with predictions on different case IDs
    root = build_run(tmp_path)
    y = np.repeat(np.arange(7), 3)
    write_predictions(root / "roi" / "vgg16" / "test_predictions.csv", "test", SEVEN, y, y, cases=np.arange(21) + 100)
    # Pairing must fail and name the scenario that does not match
    with pytest.raises(ValueError, match="roi/vgg16"):
        load_pairs(root)


def test_missing_scenario_is_an_error(tmp_path):
    # Delete one full-path scenario
    root = build_run(tmp_path)
    shutil.rmtree(root / "full" / "cnn_svm")
    # The summary must refuse to run with a scenario missing, naming it
    with pytest.raises(FileNotFoundError, match="cnn_svm"):
        summarize_main(["--experiments-root", str(root), "--bootstrap", "20"])
