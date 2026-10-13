"""T12b cross-validation: folds, CV mode in training, plan and summary (synthetic data, CPU)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.create_cv_folds import main as folds_main
from src.evaluation.cv_summary import main as cv_summary_main
from src.evaluation.evaluate import FAMILY_ORDER, SCENARIO_IDS
from src.evaluation.summarize import POOLED
from src.training.cnn_svm import main as svm_main
from src.training.cv_plan import plan, scenario_id
from src.training.ensemble import main as ensemble_main
from src.training.train import main as train_main
from tests.synthetic_data import EXPECTED_PREDICTION_COLUMNS, RAW_FOLDERS, build_synthetic_dataset
from tests.test_cnn_svm import svm_args
from tests.test_evaluation import SEVEN, write_predictions
from tests.test_training import run_args


FOLDS = 2  # the synthetic data has 4 ROI development cases per class


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return build_synthetic_dataset(tmp_path_factory.mktemp("wounds"))


@pytest.fixture(scope="module")
def folds(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("folds")
    args = ["--split-manifest", str(data["full_manifest"]), "--roi-manifest", str(data["roi_manifest"]),
            "--output-dir", str(out), "--folds", str(FOLDS), "--trials", "500"]
    assert folds_main(args) == 0
    return out / "cv_folds.csv"


def cv_args(folds, k):
    return ["--cv-folds", str(folds), "--fold", str(k)]


def test_folds_cover_development_cases_once(data, folds, tmp_path):
    table = pd.read_csv(folds, dtype=str)
    full = pd.read_csv(data["full_manifest"], dtype=str)
    roi = pd.read_csv(data["roi_manifest"], dtype=str)
    dev = set(full.loc[full["split"] != "test", "pseudonimized_id"])
    assert set(table["pseudonimized_id"]) == dev and not table["pseudonimized_id"].duplicated().any()
    assert not dev & set(full.loc[full["split"] == "test", "pseudonimized_id"])
    joined = roi[roi["split"] != "test"].merge(table, on="pseudonimized_id")
    assert (joined.groupby("fold")["canonical_wound_label"].nunique() == len(RAW_FOLDERS)).all()

    again = tmp_path / "again"
    assert folds_main(["--split-manifest", str(data["full_manifest"]), "--roi-manifest", str(data["roi_manifest"]),
                       "--output-dir", str(again), "--folds", str(FOLDS), "--trials", "500"]) == 0
    assert (again / "cv_folds.csv").read_bytes() == folds.read_bytes()


def test_cv_mode_predicts_the_held_out_fold_with_the_last_epoch(data, folds, tmp_path):
    args = run_args("mobilenet_v3_small", "roi", data, tmp_path, *cv_args(folds, 1),
                    "--max-epochs", "2", "--patience", "2", "--checkpoint", "last")
    assert train_main(args) == 0
    assert not (tmp_path / "test_predictions.csv").exists()
    preds = pd.read_csv(tmp_path / "validation_predictions.csv", dtype=str)
    fold = pd.read_csv(folds, dtype=str).set_index("pseudonimized_id")["fold"]
    assert set(preds["pseudonimized_id"].map(fold)) == {"1"}
    assert preds.columns.tolist() == EXPECTED_PREDICTION_COLUMNS
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["images_per_split"]["test"] == 0 and config["skip_test"] is True
    assert "Wrote predictions from last epoch 2" in (tmp_path / "train.log").read_text()
    assert len(pd.read_csv(tmp_path / "training_history.csv")) == 2


def test_cv_flags_go_together(data, tmp_path):
    with pytest.raises(SystemExit):
        train_main(run_args("alexnet", "roi", data, tmp_path, "--fold", "1"))


def test_cnn_svm_fixed_setting_in_cv_mode(data, folds, tmp_path):
    main_config = tmp_path / "main_run_config.json"
    main_config.write_text(json.dumps({"best_params": {"kernel": "linear", "C": 0.1}}))
    out = tmp_path / "svm"
    args = [a for a in svm_args(data, out) if a != "--smoke"] + cv_args(folds, 2) + ["--svm-params-from", str(main_config)]
    assert svm_main(args) == 0
    assert (out / "validation_predictions.csv").is_file()
    assert not (out / "test_predictions.csv").exists() and not (out / "svm_validation_tuning.csv").exists()
    assert json.loads((out / "run_config.json").read_text())["best_params"] == {"kernel": "linear", "C": 0.1}


def test_ensemble_validation_only(data, folds, tmp_path):
    members = []
    for k, seed in enumerate((43, 44, 45), start=1):
        member = tmp_path / f"member_0{k}"
        assert train_main(run_args("alexnet", "roi", data, member, *cv_args(folds, 1), "--max-epochs", "1",
                                   "--patience", "1", "--checkpoint", "last", "--seed", str(seed))) == 0
        members.append(str(member))
    assert ensemble_main(["--members", *members, "--output-dir", str(tmp_path), "--splits", "validation"]) == 0
    assert (tmp_path / "validation_predictions.csv").is_file() and not (tmp_path / "test_predictions.csv").exists()


def fake_main_run(root: Path) -> Path:
    for family in FAMILY_ORDER:
        for path in ("full", "roi"):
            folder = root / path / family
            folder.mkdir(parents=True)
            if family == "cnn_svm":
                (folder / "run_config.json").write_text(json.dumps({"best_params": {"kernel": "rbf", "C": 10}}))
            elif family == "alexnet_ensemble":
                rows = []
                for m, (seed, epoch) in enumerate(((43, 21), (44, 22), (45, 18)), start=1):
                    member = folder / f"member_0{m}"
                    member.mkdir()
                    (member / "run_config.json").write_text(json.dumps({"lr": 3e-05}))
                    rows.append({"member": m, "seed": seed, "best_epoch": epoch, "output_dir": str(member)})
                pd.DataFrame(rows).to_csv(folder / "ensemble_members.csv", index=False)
            else:
                pd.DataFrame({"lr": ["1e-5", "3e-5", "1e-4"], "best_epoch": [40, 30, 26],
                              "selected": [False, False, True]}).to_csv(folder / "tuning_summary.csv", index=False)
    return root


def test_plan_reads_settings_from_the_main_run(tmp_path):
    jobs = plan(fake_main_run(tmp_path / "main"), tmp_path / "cv", 5)
    kinds = pd.Series([j["kind"] for j in jobs]).value_counts().to_dict()
    assert kinds == {"train": 60, "member": 30, "svm": 10, "combine": 10}
    first = next(j for j in jobs if j["family"] == "efficientnet_b0" and j["input_path"] == "full" and j["fold"] == 3)
    assert (first["lr"], first["epochs"], first["job_name"]) == ("1e-4", 26, "CV_S01_efficientnet_b0_full_f3")
    member = next(j for j in jobs if j["job_name"] == "CV_S16_alexnet_ensemble_roi_f5_m3")
    assert (member["family"], member["seed"], member["epochs"], member["lr"]) == ("alexnet", 45, 18, 3e-05)
    assert member["output_dir"].endswith("roi/alexnet_ensemble/fold_5/member_03")
    assert all(scenario_id(p, f) == sid for (p, f), sid in SCENARIO_IDS.items())


def fake_cv_run(root: Path, wrong_roi=()) -> Path:
    """Every scenario: 2 folds of 14 cases each, one image per case."""
    y = np.repeat(np.arange(7), 2)
    for family in FAMILY_ORDER:
        for path in ("full", "roi"):
            for k in (1, 2):
                pred = y.copy()
                if path == "roi":
                    for i in wrong_roi:
                        pred[i] = (y[i] + 1) % 7
                file = write_predictions(root / path / family / f"fold_{k}" / "validation_predictions.csv",
                                         "validation", SEVEN, y, pred, cases=np.arange(14) + 100 * k)
                frame = pd.read_csv(file, dtype=str)
                frame["relative_path"] = f"fold_{k}/" + frame["relative_path"]  # each image in one fold only
                frame.to_csv(file, index=False)
    return root


def test_cv_summary_identical_paths(tmp_path):
    root = fake_cv_run(tmp_path)
    assert cv_summary_main(["--cv-root", str(root), "--folds", "2", "--bootstrap", "50"]) == 0
    table = pd.read_csv(root / "summary" / "cv_table.csv")
    assert len(table) == 16 and (table["oof_cases"] == 28).all()
    assert table["macro_f1_mean"].tolist() == pytest.approx([1.0] * 16)
    comparison = pd.read_csv(root / "summary" / "cv_full_vs_roi.csv")
    assert np.allclose(comparison["difference"], 0.0)
    assert (root / "summary" / "cv_full_vs_roi_macro_f1.png").is_file()


def test_cv_summary_worse_roi_and_missing_fold(tmp_path):
    root = fake_cv_run(tmp_path / "a", wrong_roi=(0, 3, 6))
    assert cv_summary_main(["--cv-root", str(root), "--folds", "2", "--bootstrap", "50"]) == 0
    f1 = pd.read_csv(root / "summary" / "cv_full_vs_roi.csv").query("metric == 'macro_f1'").set_index("family")
    assert (f1.drop(POOLED)["difference"] < 0).all()

    (root / "roi" / "vgg16" / "fold_2" / "validation_predictions.csv").unlink()
    with pytest.raises(FileNotFoundError, match="roi/vgg16"):
        cv_summary_main(["--cv-root", str(root), "--folds", "2", "--bootstrap", "10"])
