"""T16 final verification: a hand-built finished run passes, and each kind of fault is caught (CPU)."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cv_summary import main as cv_summary_main
from src.evaluation.evaluate import FAMILY_ORDER, INPUT_PATHS, main as evaluate_main
from src.evaluation.gradcam import FAMILIES as GRADCAM_FAMILIES
from src.evaluation.metrics import compute_metrics, load_predictions
from src.evaluation.summarize import main as summarize_main
from src.evaluation.verify import ENSEMBLE_SEEDS, SINGLE_FAMILIES, main as verify_main
from src.training.select_lr import LR_GRID
from tests.test_evaluation import SEVEN, write_predictions


PER_CLASS = 3
TEST_CASES = np.arange(21)          # 3 per class
VAL_CASES = TEST_CASES + 100
TRAIN_CASES = TEST_CASES + 200
CHOSEN_LR = "3e-5"
BOOT = 30
BUDGET = {"max_epochs": 3, "patience": 2, "batch_size": 8, "weight_decay": 1e-4, "label_smoothing": 0.05,
          "full_scope": "matched", "class_weights": True, "pretrained": True,
          "git_commit": "abc1234", "git_dirty": False}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_split(folder: Path, wrong=(), split_cases=None) -> None:
    """Validation and test predictions of one model; ``wrong`` rows are misclassified on both."""
    y = np.repeat(np.arange(len(SEVEN)), PER_CLASS)
    pred = y.copy()
    for i in wrong:
        pred[i] = (y[i] + 1) % len(SEVEN)
    for split, cases in (split_cases or {"validation": VAL_CASES, "test": TEST_CASES}).items():
        write_predictions(folder / f"{split}_predictions.csv", split, SEVEN, y, pred, cases)


def write_trained(folder: Path, lr: str, seed: int, wrong, skip_test: bool, loss: float) -> float:
    """A train.py run: config, a 3-epoch history whose epoch 2 is best, validation predictions."""
    write_split(folder, wrong, {"validation": VAL_CASES})
    write_json(folder / "run_config.json", {**BUDGET, "lr": float(lr), "seed": seed, "skip_test": skip_test})
    f1 = compute_metrics(load_predictions(folder / "validation_predictions.csv", "validation"))["macro_f1"]
    pd.DataFrame({"epoch": [1, 2, 3], "validation_macro_f1": [f1 - 0.1, f1, f1 - 0.05],
                  "validation_loss": [loss + 0.2, loss, loss + 0.1]}).to_csv(folder / "training_history.csv", index=False)
    return f1


def write_tuned(folder: Path, wrong) -> None:
    """T08b layout: three grid runs, tuning_summary.csv, then test for the chosen run only."""
    rows = []
    for lr in LR_GRID:
        f1 = write_trained(folder / f"lr_{lr}", lr, 42, wrong, lr != CHOSEN_LR, 0.5 if lr == CHOSEN_LR else 0.6)
        rows.append({"lr": lr, "best_epoch": 2, "validation_macro_f1": f1,
                     "validation_loss": 0.5 if lr == CHOSEN_LR else 0.6,
                     "output_dir": str(folder / f"lr_{lr}"), "selected": lr == CHOSEN_LR})
    pd.DataFrame(rows).to_csv(folder / "tuning_summary.csv", index=False)
    chosen = folder / f"lr_{CHOSEN_LR}"
    write_split(chosen, wrong, {"test": TEST_CASES})
    config = json.loads((chosen / "run_config.json").read_text())
    write_json(chosen / "run_config.json", {**config, "skip_test": False})
    for name in ("run_config.json", "training_history.csv", "validation_predictions.csv", "test_predictions.csv"):
        shutil.copy2(chosen / name, folder / name)


def write_svm(folder: Path, wrong) -> None:
    pd.DataFrame([{"kernel": "rbf", "C": 1.0, "gamma": "scale", "validation_macro_f1": 0.4,
                   "validation_balanced_accuracy": 0.4},
                  {"kernel": "linear", "C": 0.1, "gamma": "", "validation_macro_f1": 0.7,
                   "validation_balanced_accuracy": 0.7}]).to_csv(folder / "svm_validation_tuning.csv", index=False)
    write_json(folder / "run_config.json", {"family": "cnn_svm", "best_params": {"kernel": "linear", "C": 0.1},
                                            "best_validation_macro_f1": 0.7})
    write_split(folder, wrong)


def write_ensemble(folder: Path, wrong) -> None:
    rows = []
    for j, seed in enumerate(ENSEMBLE_SEEDS, start=1):
        member = folder / f"member_{j:02d}"
        f1 = write_trained(member, CHOSEN_LR, seed, wrong, False, 0.5)
        write_split(member, wrong, {"test": TEST_CASES})
        rows.append({"member": j, "seed": seed, "best_epoch": 2, "best_validation_macro_f1": f1,
                     "output_dir": str(member), "representative": j == 1})
    pd.DataFrame(rows).to_csv(folder / "ensemble_members.csv", index=False)
    write_json(folder / "run_config.json", {"family": "ensemble", "seeds": list(ENSEMBLE_SEEDS)})
    write_split(folder, wrong)


def write_gradcam(root: Path) -> None:
    out = root / "gradcam"
    out.mkdir()
    examples = pd.DataFrame([{"pseudonimized_id": str(c * PER_CLASS + k), "canonical_wound_label": name,
                              "split": "test", "set": k + 1}
                             for c, name in enumerate(SEVEN) for k in range(2)])
    examples.to_csv(out / "examples.csv", index=False)
    rows = []
    for family in GRADCAM_FAMILIES:
        for input_path in INPUT_PATHS:
            model = f"lr_{CHOSEN_LR}" if family in SINGLE_FAMILIES else "member_01"
            for e in examples.to_dict("records"):
                rows.append({"family": family, "input_path": input_path, "set": e["set"],
                             "class": e["canonical_wound_label"], "case": e["pseudonimized_id"],
                             "image": f"test/img_{e['pseudonimized_id']}.jpg", "matches_prediction_file": True,
                             "checkpoint": str(root / input_path / family / model / "best.pt")})
            for k in (1, 2):
                shutil.copy(root / "full" / "vgg16" / "test_confusion_matrix.png", out / f"{family}_set{k}.png")
    pd.DataFrame(rows).to_csv(out / "gradcam_results.csv", index=False)


def write_cv(cv_root: Path, data_dir: Path) -> Path:
    """Two folds: validation cases are fold 1, train cases fold 2."""
    folds = pd.DataFrame({"pseudonimized_id": [str(c) for c in np.concatenate([VAL_CASES, TRAIN_CASES])],
                          "fold": [1] * len(VAL_CASES) + [2] * len(TRAIN_CASES)})
    folds.to_csv(data_dir / "cv_folds.csv", index=False)
    for family in FAMILY_ORDER:
        for input_path, wrong in (("full", (0, 4, 9)), ("roi", (4,))):
            for k, cases in ((1, VAL_CASES), (2, TRAIN_CASES)):
                fold = cv_root / input_path / family / f"fold_{k}"
                write_split(fold, wrong, {"validation": cases})
                path = fold / "validation_predictions.csv"
                frame = pd.read_csv(path, dtype=str)
                frame["relative_path"] = f"fold_{k}/" + frame["relative_path"]
                frame.to_csv(path, index=False)
                write_json(fold / "run_config.json", {"family": family, "cv_folds": "cv_folds.csv", "fold": k})
    assert cv_summary_main(["--cv-root", str(cv_root), "--folds", "2", "--bootstrap", str(BOOT)]) == 0
    return data_dir / "cv_folds.csv"


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    base = tmp_path_factory.mktemp("t16")
    root, data_dir = base / "main", base / "data"
    data_dir.mkdir()
    pd.DataFrame({"pseudonimized_id": [str(c) for c in np.concatenate([TEST_CASES, VAL_CASES, TRAIN_CASES])],
                  "split": ["test"] * 21 + ["validation"] * 21 + ["train"] * 21}) \
        .to_csv(data_dir / "assignments.csv", index=False)
    for family in FAMILY_ORDER:
        for input_path, wrong in (("full", (0, 4, 9)), ("roi", (4,))):
            folder = root / input_path / family
            folder.mkdir(parents=True)
            {"cnn_svm": write_svm, "alexnet_ensemble": write_ensemble}.get(family, write_tuned)(folder, wrong)
    assert evaluate_main(["--experiments-root", str(root), "--bootstrap", str(BOOT)]) == 0
    assert summarize_main(["--experiments-root", str(root), "--bootstrap", str(BOOT)]) == 0
    write_gradcam(root)
    cv_folds = write_cv(base / "cv", data_dir)
    return base, cv_folds


def run_verify(base: Path, cv_folds: Path, with_cv: bool = True) -> pd.DataFrame:
    args = ["--experiments-root", str(base / "main"), "--split-assignments", str(base / "data" / "assignments.csv"),
            "--bootstrap", str(BOOT)]
    if with_cv:
        args += ["--cv-root", str(base / "cv"), "--cv-folds", str(cv_folds), "--folds", "2"]
    code = verify_main(args)
    report = pd.read_csv(base / "main" / "verification" / "verification_report.csv", keep_default_na=False)
    assert code == (1 if (report["status"] == "FAIL").any() else 0)
    return report


@pytest.fixture
def copy(clean, tmp_path):
    base, cv_folds = clean
    shutil.copytree(base, tmp_path / "run", copy_function=shutil.copy2)
    return tmp_path / "run", tmp_path / "run" / "data" / "cv_folds.csv"


def failures(report: pd.DataFrame) -> pd.DataFrame:
    return report[report["status"] == "FAIL"]


def test_clean_run_passes(clean):
    report = run_verify(*clean)
    assert failures(report).empty, failures(report).to_string()
    assert set(report["check"]) == {"files", "test set", "selection", "budget", "numbers", "gradcam", "cv"}
    assert (report["check"] == "files").sum() == 16 + 4


def test_test_predictions_of_an_unchosen_lr_fail(copy):
    base, cv_folds = copy
    shutil.copy2(base / "main/roi/vgg16/test_predictions.csv", base / "main/roi/vgg16/lr_1e-4/test_predictions.csv")
    bad = failures(run_verify(base, cv_folds, with_cv=False))
    assert bad["item"].tolist() == ["S10 roi/vgg16: lr_1e-4 never predicted test"]


def test_wrong_lr_choice_fails(copy):
    base, cv_folds = copy
    path = base / "main/full/resnet152/tuning_summary.csv"
    table = pd.read_csv(path, dtype={"lr": str})
    table["selected"] = table["lr"] == "1e-4"
    table.to_csv(path, index=False)
    bad = failures(run_verify(base, cv_folds, with_cv=False))
    assert bad["item"].tolist() == ["S03 full/resnet152: chosen lr reproduced",
                                    "S03 full/resnet152: test predicted after the lr choice"]  # table rewritten after test


def test_edited_summary_number_fails(copy):
    base, cv_folds = copy
    path = base / "main/summary/full_vs_roi.csv"
    table = pd.read_csv(path)
    table.loc[0, "difference"] += 0.01
    table.to_csv(path, index=False)
    bad = failures(run_verify(base, cv_folds, with_cv=False))
    assert bad["item"].tolist() == ["full_vs_roi.csv"]


def test_ensemble_not_the_mean_of_its_members_fails(copy):
    base, cv_folds = copy
    folder = base / "main/full/alexnet_ensemble"
    write_split(folder / "member_02", wrong=(1, 2), split_cases={"test": TEST_CASES})
    bad = failures(run_verify(base, cv_folds, with_cv=False))
    assert "S15 full/alexnet_ensemble: test = mean of members" in bad["item"].tolist()


def test_missing_figure_fails(copy):
    base, cv_folds = copy
    (base / "main/roi/googlenet/test_roc_curves.png").unlink()
    (base / "main/gradcam/vit_b16_set2.png").unlink()
    bad = failures(run_verify(base, cv_folds, with_cv=False))
    assert bad["item"].tolist() == ["S08 roi/googlenet", "figures"]


def test_test_case_in_training_split_fails(copy):
    base, cv_folds = copy
    path = base / "data/assignments.csv"
    table = pd.read_csv(path, dtype=str)
    table.loc[table["pseudonimized_id"] == "4", "split"] = "train"
    table.to_csv(path, index=False)
    bad = failures(run_verify(base, cv_folds, with_cv=False))
    assert len(bad) == 16 + 1 and set(bad["check"]) == {"test set", "gradcam"}


def test_cv_run_with_test_predictions_fails(copy):
    base, cv_folds = copy
    shutil.copy2(base / "main/full/vgg16/test_predictions.csv", base / "cv/full/vgg16/fold_1/test_predictions.csv")
    bad = failures(run_verify(base, cv_folds))
    assert bad["item"].tolist() == ["no CV run predicted test"]
