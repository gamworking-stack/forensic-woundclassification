"""T17 per-box ROI experiment: photo weights, photo-level selection and summary (synthetic, CPU)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from src.evaluation.box_summary import main as box_summary_main
from src.evaluation.evaluate import FAMILY_ORDER, main as evaluate_main
from src.evaluation.metrics import load_predictions
from src.training.cnn_svm import main as svm_main
from src.training.engine import classification_metrics, load_checkpoint, weighted_loss
from src.training.ensemble import soft_vote
from src.training.photo import PHOTO_COLUMN, manifest_photos, photo_mean, photo_prediction_frame, photo_weights
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


def test_manifest_photos_reads_relative_path_and_rejects_empty():
    rows = pd.DataFrame({"relative_path": ["cut/a.JPG", "cut/a.JPG"], "source_relative_path": ["", ""]})
    assert manifest_photos(rows).tolist() == ["cut/a.JPG", "cut/a.JPG"]
    with pytest.raises(ValueError, match="empty"):
        manifest_photos(pd.DataFrame({"relative_path": ["cut/a.JPG", " "]}))


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
        assert set(preds[PHOTO_COLUMN]) == set(manifest.loc[manifest["split"] == split, "relative_path"])
        assert preds[PHOTO_COLUMN].notna().all()

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


# ---------- cross-validation plan and summary ----------

def test_cv_plan_for_box_reads_the_per_box_run(tmp_path):
    from src.training.cv_plan import plan
    from tests.test_cv import fake_main_run
    run = fake_main_run(tmp_path / "roi_box")
    (run / "roi").rename(run / "box")  # the per-box run has the main run's layout under box/
    members = run / "box" / "alexnet_ensemble" / "ensemble_members.csv"
    table = pd.read_csv(members)
    table["output_dir"] = table["output_dir"].str.replace("/roi/", "/box/")
    table.to_csv(members, index=False)
    jobs = plan(run, tmp_path / "cv_box", 5, ("box",))
    kinds = pd.Series([j["kind"] for j in jobs]).value_counts().to_dict()
    assert kinds == {"train": 30, "member": 15, "svm": 5, "combine": 5}
    assert jobs[0]["job_name"] == "CV_S17_efficientnet_b0_box_f1" and jobs[0]["input_path"] == "box"
    assert any(j["job_name"] == "CV_S24_alexnet_ensemble_box_f5_m3" for j in jobs)


def write_box_fold(path: Path, cases, y, wrong=(), crops_per_photo=2):
    """Crop-level held-out-fold file: ``crops_per_photo`` crops per case, one photo per case."""
    y_crop, pred = np.repeat(y, crops_per_photo), np.repeat(y, crops_per_photo)
    for i in wrong:
        pred[i * crops_per_photo:(i + 1) * crops_per_photo] = (y[i] + 1) % 7
    file = write_predictions(path, "validation", SEVEN, y_crop, pred, np.repeat(cases, crops_per_photo))
    frame = pd.read_csv(file, dtype=str)
    frame["relative_path"] = path.parent.name + "/" + frame["relative_path"]
    frame.insert(1, PHOTO_COLUMN, [f"photo_{c}.jpg" for c in np.repeat(cases, crops_per_photo)])
    frame.to_csv(file, index=False)


def test_cv_box_summary_against_t12b(tmp_path):
    from src.evaluation.cv_box_summary import main as cv_box_main
    from src.evaluation.cv_summary import main as cv_summary_main
    from tests.test_cv import fake_cv_run
    main_cv = fake_cv_run(tmp_path / "cv", wrong_roi=(0, 3))
    assert cv_summary_main(["--cv-root", str(main_cv), "--folds", "2", "--bootstrap", "20"]) == 0
    box_cv = tmp_path / "cv_box"
    y = np.repeat(np.arange(7), 2)
    for family in FAMILY_ORDER:
        for k in (1, 2):
            write_box_fold(box_cv / "box" / family / f"fold_{k}" / "validation_predictions.csv", np.arange(14) + 100 * k, y)
    assert cv_box_main(["--cv-box-root", str(box_cv), "--cv-main-root", str(main_cv), "--folds", "2",
                        "--bootstrap", "30"]) == 0
    table = pd.read_csv(box_cv / "summary" / "cv_box_table.csv")
    assert table["scenario_id"].tolist() == [f"S{i}" for i in range(17, 25)]
    assert (table["oof_photos"] == 28).all() and table["macro_f1_mean"].tolist() == pytest.approx([1.0] * 8)
    union = pd.read_csv(box_cv / "summary" / "cv_box_vs_union.csv").query("metric == 'macro_f1'").set_index("family")
    assert (union.drop("all families (mean)")["difference"] > 0).all()  # union had 2 wrong per fold
    full = pd.read_csv(box_cv / "summary" / "cv_box_vs_full.csv").query("metric == 'macro_f1'")
    assert np.allclose(full["difference"], 0.0)


# ---------- Grad-CAM on per-box crops ----------

def test_gradcam_box_explains_every_crop_of_the_main_examples(data, tmp_path):
    from src.evaluation.gradcam import draw_examples, main as gradcam_main
    exp = tmp_path / "roi_box"
    assert train_main(["--family", "mobilenet_v3_small", *box_args(data, exp / "box" / "mobilenet_v3_small", "--max-epochs", "1")]) == 0
    member = exp / "box" / "alexnet_ensemble" / "member_01"
    assert train_main(["--family", "alexnet", *box_args(data, member, "--max-epochs", "1", "--seed", "43")]) == 0
    pd.DataFrame([{"member": 1, "seed": 43, "best_epoch": 1, "best_validation_macro_f1": 0.5,
                   "output_dir": str(member), "representative": True}]) \
        .to_csv(exp / "box" / "alexnet_ensemble" / "ensemble_members.csv", index=False)
    examples = tmp_path / "examples.csv"
    draw_examples(data["roi_manifest"], 2, 42).to_csv(examples, index=False)

    assert gradcam_main(["--experiments-root", str(exp), "--box-manifest", str(data["box_manifest"]),
                         "--examples", str(examples), "--families", "mobilenet_v3_small", "alexnet_ensemble",
                         "--no-pretrained", "--device", "cpu"]) == 0
    results = pd.read_csv(exp / "gradcam" / "gradcam_results.csv")
    crops = pd.read_csv(data["box_manifest"], dtype=str)
    n_crops = crops["relative_path"].isin(pd.read_csv(examples, dtype=str)["relative_path"]).sum()
    assert len(results) == 2 * n_crops and n_crops > 14  # some photos have several crops
    assert results["matches_prediction_file"].all() and results["matches_photo_prediction_file"].all()
    for family in ("mobilenet_v3_small", "alexnet_ensemble"):
        for k in (1, 2):
            figure = np.asarray(Image.open(exp / "gradcam" / f"{family}_set{k}.png").convert("L"))
            assert (figure < 250).mean() > 0.05  # crops are drawn, not just the title (examples.csv holds set as text)


# ---------- verification of the per-box run ----------

from tests.test_verify import (  # noqa: E402  (fixture and writers of the T16 tests)
    BOOT, TEST_CASES, TRAIN_CASES, VAL_CASES, clean, write_ensemble, write_split, write_svm, write_tuned,
)


def add_photo_column(root: Path) -> None:
    """One crop per photo: every prediction file names its photo; every run selects on photo level."""
    for path in sorted(root.rglob("*_predictions.csv")):
        frame = pd.read_csv(path, dtype=str)
        if PHOTO_COLUMN not in frame.columns:
            frame.insert(1, PHOTO_COLUMN, "photo_" + frame["pseudonimized_id"] + ".jpg")
            frame.to_csv(path, index=False)
    for path in sorted(root.rglob("run_config.json")):
        config = json.loads(path.read_text())
        if config.get("family") not in ("cnn_svm", "ensemble"):
            path.write_text(json.dumps({**config, "selection_unit": "photo"}))


@pytest.fixture(scope="module")
def box_run(clean, tmp_path_factory):
    import shutil
    from src.evaluation.cv_box_summary import main as cv_box_main
    base, cv_folds = clean
    work = tmp_path_factory.mktemp("t17v")
    shutil.copytree(base, work / "run", copy_function=shutil.copy2)
    base = work / "run"
    root = base / "roi_box"
    for family in FAMILY_ORDER:
        folder = root / "box" / family
        folder.mkdir(parents=True)
        {"cnn_svm": write_svm, "alexnet_ensemble": write_ensemble}.get(family, write_tuned)(folder, ())
    add_photo_column(root)
    assert evaluate_main(["--experiments-root", str(root), "--bootstrap", str(BOOT)]) == 0
    assert box_summary_main(["--box-root", str(root), "--main-root", str(base / "main"), "--bootstrap", str(BOOT)]) == 0
    for family in FAMILY_ORDER:
        for k, cases in ((1, VAL_CASES), (2, TRAIN_CASES)):
            fold = base / "cv_box" / "box" / family / f"fold_{k}"
            write_split(fold, (), {"validation": cases})
            frame = pd.read_csv(fold / "validation_predictions.csv", dtype=str)
            frame["relative_path"] = f"fold_{k}/" + frame["relative_path"]
            frame.to_csv(fold / "validation_predictions.csv", index=False)
            (fold / "run_config.json").write_text(json.dumps({"family": family, "cv_folds": "cv_folds.csv", "fold": k}))
    add_photo_column(base / "cv_box")
    assert cv_box_main(["--cv-box-root", str(base / "cv_box"), "--cv-main-root", str(base / "cv"), "--folds", "2",
                        "--bootstrap", str(BOOT)]) == 0
    return base, base / "data" / "cv_folds.csv"


def run_verify_box(base: Path, cv_folds: Path) -> pd.DataFrame:
    from src.evaluation.verify_box import main as verify_box_main
    code = verify_box_main(["--box-root", str(base / "roi_box"), "--main-root", str(base / "main"),
                            "--split-assignments", str(base / "data" / "assignments.csv"),
                            "--cv-root", str(base / "cv_box"), "--cv-main-root", str(base / "cv"),
                            "--cv-folds", str(cv_folds), "--folds", "2", "--bootstrap", str(BOOT)])
    report = pd.read_csv(base / "roi_box" / "verification" / "verification_report.csv", keep_default_na=False)
    assert code == (1 if (report["status"] == "FAIL").any() else 0)
    return report


def test_verify_box_clean_run_passes(box_run):
    report = run_verify_box(*box_run)
    bad = report[report["status"] == "FAIL"]
    assert bad.empty, bad.to_string()
    assert set(report["check"]) == {"files", "test set", "selection", "budget", "numbers", "gradcam", "cv"}


def test_verify_box_catches_an_edited_number(box_run, tmp_path):
    import shutil
    base, _ = box_run
    shutil.copytree(base, tmp_path / "run", copy_function=shutil.copy2)
    path = tmp_path / "run" / "roi_box" / "summary" / "box_vs_union.csv"
    table = pd.read_csv(path)
    table.loc[0, "difference"] += 0.01
    table.to_csv(path, index=False)
    report = run_verify_box(tmp_path / "run", tmp_path / "run" / "data" / "cv_folds.csv")
    assert report.loc[report["status"] == "FAIL", "item"].tolist() == ["box_vs_union.csv"]
