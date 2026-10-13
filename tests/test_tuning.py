"""T08b learning-rate tuning: validation-only grid runs, selection, then one test prediction."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
import torch

from src.evaluation.evaluate import find_scenarios
from src.training.select_lr import choose, main as select_main
from src.training.train import main as train_main
from tests.synthetic_data import EXPECTED_PREDICTION_COLUMNS, build_synthetic_dataset
from tests.test_training import run_args


GRID = ("1e-5", "3e-5", "1e-4")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return build_synthetic_dataset(tmp_path_factory.mktemp("wounds"))


@pytest.fixture(scope="module")
def tuned_scenario(dataset, tmp_path_factory):
    """Three --skip-test grid runs in <run>/roi/mobilenet_v3_small/lr_<value>/."""
    root = tmp_path_factory.mktemp("run")
    scenario = root / "roi" / "mobilenet_v3_small"
    for lr in GRID:
        args = run_args("mobilenet_v3_small", "roi", dataset, scenario / f"lr_{lr}",
                        "--smoke", "--skip-test", "--lr", lr)
        assert train_main(args) == 0
    return root, scenario


def grid_run(lr, f1, loss):
    return {"lr": lr, "validation_macro_f1": f1, "validation_loss": loss}


def test_skip_test_then_resume_adds_test_without_training(dataset, tmp_path):
    base = run_args("mobilenet_v3_small", "roi", dataset, tmp_path, "--smoke")
    assert train_main(base + ["--skip-test"]) == 0
    assert (tmp_path / "validation_predictions.csv").is_file()
    assert not (tmp_path / "test_predictions.csv").exists()

    assert train_main(base + ["--resume"]) == 0
    assert (tmp_path / "test_predictions.csv").is_file()
    assert len(pd.read_csv(tmp_path / "training_history.csv")) == 2
    assert "Training already finished" in (tmp_path / "train.log").read_text()


def test_choose_prefers_f1_then_loss_then_smaller_lr():
    assert choose([grid_run("1e-5", 0.5, 1.0), grid_run("3e-5", 0.7, 1.0), grid_run("1e-4", 0.6, 0.1)]) == 1
    assert choose([grid_run("1e-5", 0.7, 1.0), grid_run("3e-5", 0.7, 0.9), grid_run("1e-4", 0.6, 0.1)]) == 1
    assert choose([grid_run("1e-5", 0.7, 0.9), grid_run("3e-5", 0.7, 0.9), grid_run("1e-4", 0.7, 0.9)]) == 0


def test_missing_grid_run_is_an_error(tuned_scenario):
    with pytest.raises(FileNotFoundError, match="lr_3e-4"):
        select_main(["--scenario-dir", str(tuned_scenario[1]), "--lrs", "1e-5", "3e-4", "--device", "cpu"])


def test_mismatched_learning_rate_is_an_error(tuned_scenario, tmp_path):
    scenario = tmp_path / "scenario"
    for name in ("best.pt", "validation_predictions.csv", "run_config.json"):
        (scenario / "lr_1e-4").mkdir(parents=True, exist_ok=True)
        (scenario / "lr_1e-4" / name).write_bytes((tuned_scenario[1] / "lr_1e-5" / name).read_bytes())
    with pytest.raises(ValueError, match="lr=1e-05, expected 1e-4"):
        select_main(["--scenario-dir", str(scenario), "--lrs", "1e-4", "--device", "cpu"])


def test_select_fills_scenario_folder_from_the_chosen_run(tuned_scenario):
    root, scenario = tuned_scenario
    assert select_main(["--scenario-dir", str(scenario), "--device", "cpu", "--num-workers", "0"]) == 0

    summary = pd.read_csv(scenario / "tuning_summary.csv", dtype={"lr": str})
    assert summary["lr"].tolist() == list(GRID)
    assert summary["selected"].sum() == 1
    chosen = summary.loc[summary["selected"], "lr"].item()
    for row in summary.to_dict("records"):
        best = torch.load(scenario / f"lr_{row['lr']}" / "best.pt", map_location="cpu", weights_only=False)
        assert row["best_epoch"] == best["epoch"]
        assert row["validation_macro_f1"] == pytest.approx(best["validation_macro_f1"])
    for lr in GRID:
        assert (scenario / f"lr_{lr}" / "test_predictions.csv").exists() == (lr == chosen)

    for name in ("run_config.json", "training_history.csv", "validation_predictions.csv", "test_predictions.csv"):
        assert (scenario / name).read_bytes() == (scenario / f"lr_{chosen}" / name).read_bytes()
    test = pd.read_csv(scenario / "test_predictions.csv")
    assert test.columns.tolist() == EXPECTED_PREDICTION_COLUMNS
    assert (test["split"] == "test").all()
    config = json.loads((scenario / "run_config.json").read_text())
    assert config["lr"] == float(chosen) and config["skip_test"] is False

    log = (scenario / "tuning.log").read_text()
    assert log.index("Selected lr") < log.index("Copied")
    assert [(p, f) for p, f, _ in find_scenarios(root)] == [("roi", "mobilenet_v3_small")]

    before = {name: (scenario / name).read_bytes() for name in ("tuning_summary.csv", "test_predictions.csv")}
    assert select_main(["--scenario-dir", str(scenario), "--device", "cpu", "--num-workers", "0"]) == 0
    assert {name: (scenario / name).read_bytes() for name in before} == before
    assert len(pd.read_csv(scenario / f"lr_{chosen}" / "training_history.csv")) == 2

