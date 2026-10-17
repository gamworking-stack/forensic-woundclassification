"""T19 auto-box ROI pipeline on synthetic data (CPU, no downloads): selection, detector arms with fake or
untrained models, auto crops, existing models on other crops, the comparisons and parts of verification."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("pycocotools")

from src.data.create_auto_roi_dataset import main as crops_main  # noqa: E402
from src.detection import finetune, public, zero_shot  # noqa: E402
from src.detection.boxes import padded_crop_box, training_limits  # noqa: E402
from src.detection.select import main as select_main  # noqa: E402
from src.evaluation.evaluate import arm_scenario_ids  # noqa: E402
from tests.synthetic_data import RAW_FOLDERS  # noqa: E402

LABELS = ("cut wound", "scratch")
SIZE = (200, 150)


# ---------- a small world: raw photos, manual boxes, manifests ----------

@pytest.fixture()
def world(tmp_path):
    # Three photos per class and split, each its own case; seeded so every test sees the same world
    rng = np.random.default_rng(0)
    crops, reference, folds = [], [], []
    case = 0
    for split in ("train", "validation", "test"):
        for label in LABELS:
            for k in range(3):
                case += 1
                # Save a random raw photo in the raw folder of its wound type
                stem = f"IMG_{case:04d}"
                folder = RAW_FOLDERS[label]
                (tmp_path / "raw" / folder).mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)).save(tmp_path / "raw" / folder / f"{stem}.JPG")
                # One manual box per photo, shifted by k; the first photo of each class gets a second box
                boxes = [[20 + 10 * k, 20, 60 + 10 * k, 50]] + ([[120, 80, 170, 130]] if k == 0 else [])
                photo = f"{label}/{stem}.JPG"
                # Each box becomes a padded crop row (box manifest) and an unpadded reference row
                for i, box in enumerate(boxes, start=1):
                    x1, y1, x2, y2 = padded_crop_box([box], *SIZE)
                    crops.append({"pseudonimized_id": str(case), "wound_id": "1.0", "filename": f"{stem}.JPG",
                                  "wound_label": folder, "canonical_wound_label": label, "relative_path": photo,
                                  "split": split, "roi_relative_path": f"{split}/{label}/{stem}_bbox{i:02d}.jpg",
                                  "crop_index": i, "padding_ratio": 0.08, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                  "source_width": SIZE[0], "source_height": SIZE[1]})
                    reference.append({"relative_path": photo, "pseudonimized_id": str(case), "split": split,
                                      "canonical_wound_label": label, "box_index": i, "x1": box[0], "y1": box[1],
                                      "x2": box[2], "y2": box[3], "source_width": SIZE[0], "source_height": SIZE[1],
                                      "annotation_export": "task.zip"})
                # Training cases get folds 1-3 for the 4F cross-fitting
                if split == "train":
                    folds.append({"pseudonimized_id": str(case), "fold": str(1 + k)})
    # Write the manifest, reference boxes, limits from training boxes, and the CV folds
    pd.DataFrame(crops).to_csv(tmp_path / "box_manifest.csv", index=False)
    ref = pd.DataFrame(reference)
    ref.to_csv(tmp_path / "reference.csv", index=False)
    (tmp_path / "limits.json").write_text(json.dumps(training_limits(ref)))
    pd.DataFrame(folds).to_csv(tmp_path / "cv_folds.csv", index=False)
    return {"root": tmp_path, "raw": tmp_path / "raw", "box_manifest": tmp_path / "box_manifest.csv",
            "reference": ref, "reference_csv": tmp_path / "reference.csv", "limits": tmp_path / "limits.json",
            "cv_folds": tmp_path / "cv_folds.csv"}


def good_detections(reference, splits, skip=(), low=()):
    """Jittered manual boxes (score 0.9, or 0.05 for photos in ``low``) plus one stray box per photo."""
    # Near-perfect detections: every manual box shrunk by a pixel, unless the photo is skipped
    rows = []
    for photo, group in reference[reference["split"].isin(splits)].groupby("relative_path"):
        if photo in skip:
            continue
        score = 0.05 if photo in low else 0.9
        for r in group.to_dict("records"):
            rows.append([photo, score, r["x1"] + 1, r["y1"] + 1, r["x2"] - 1, r["y2"] - 1])
        # Plus one low-scoring false box in the corner of every photo
        rows.append([photo, 0.1, 0, 0, 15, 15])
    return pd.DataFrame(rows, columns=["relative_path", "score", "x1", "y1", "x2", "y2"])


def random_detections(reference, splits, seed=1):
    # One random-scored box per photo in an empty corner, so it never matches a manual box
    rng = np.random.default_rng(seed)
    rows = [[p, float(rng.uniform()), 150, 0, 199, 20] for p in sorted(set(reference.loc[reference["split"].isin(splits), "relative_path"]))]
    return pd.DataFrame(rows, columns=["relative_path", "score", "x1", "y1", "x2", "y2"])


def select_args(w, dev, out, *configs, arm="4Z"):
    return ["--arm", arm, "--detections-dir", str(dev), "--configs", *configs, "--output", str(out),
            "--reference", str(w["reference_csv"]), "--limits", str(w["limits"])]


# ---------- selection ----------

def test_select_picks_the_best_drops_test_rows_and_is_written_once(world):
    # Two configs: a useless one and a near-perfect one that also has test rows
    dev = world["root"] / "dev"
    dev.mkdir()
    random_detections(world["reference"], ("train", "validation")).to_csv(dev / "bad.csv", index=False)
    good = good_detections(world["reference"], ("train", "validation", "test"))  # test rows must be ignored
    good.to_csv(dev / "good.csv", index=False)
    # Select on train + validation for arm 4Z
    out = world["root"] / "selection.json"
    assert select_main(select_args(world, dev, out, "bad", "good")) == 0
    # The good config wins and passes the gate, with the threshold at its true-box score
    chosen = json.loads(out.read_text())
    assert chosen["chosen_config"] == "good" and chosen["gate_passed"]
    assert chosen["threshold"] == pytest.approx(0.9) and chosen["threshold_detail"]["recall"] == pytest.approx(1.0)
    # All of its test rows were dropped before selection, and only the 12 development photos were used
    assert chosen["test_detections_dropped"]["good"] == int((good["relative_path"].isin(
        world["reference"].loc[world["reference"]["split"] == "test", "relative_path"])).sum())
    assert chosen["selection_photos"] == 12  # train + validation for 4Z
    # selection.json is frozen: a second selection must refuse to overwrite it
    with pytest.raises(SystemExit, match="frozen"):
        select_main(select_args(world, dev, out, "bad", "good"))


def test_select_ties_go_to_the_first_config_and_4f_selects_on_validation(world):
    # Two configs with identical detections on validation only
    dev = world["root"] / "dev"
    dev.mkdir()
    for name in ("a", "b"):
        good_detections(world["reference"], ("validation",)).to_csv(dev / f"{name}.csv", index=False)
    # Arm 4F selects on validation, and the tie goes to the config listed first
    out = world["root"] / "sel.json"
    assert select_main(select_args(world, dev, out, "b", "a", arm="4F")) == 0
    chosen = json.loads(out.read_text())
    assert chosen["chosen_config"] == "b" and chosen["selection_splits"] == ["validation"] and chosen["selection_photos"] == 6


def test_gate_fails_for_a_detector_that_finds_little(world):
    # Only a detector that never finds a manual box
    dev = world["root"] / "dev"
    dev.mkdir()
    random_detections(world["reference"], ("train", "validation")).to_csv(dev / "bad.csv", index=False)
    out = world["root"] / "sel.json"
    # Its recall is far below 0.50, so the gate must fail
    select_main(select_args(world, dev, out, "bad"))
    assert not json.loads(out.read_text())["gate_passed"]


# ---------- 4Z ----------

def test_zero_shot_configs_prompts_and_tiles():
    # 16 configs: 2 models x 4 prompt sets x plain/tiled, interleaved in a fixed order
    order = zero_shot.config_order()
    assert len(order) == 16 and order[:4] == ["gdino_P1", "owlv2_P1", "gdino_P1_tiled", "owlv2_P1_tiled"]
    # Grounding DINO text is lower-case phrases ending in ' .'; P3 and P4 have one phrase per wound type
    assert zero_shot.grounding_text(["Cut wound", "scratch"]) == "cut wound . scratch ."
    assert len(zero_shot.PROMPTS["P3"]) == 7 and len(zero_shot.PROMPTS["P4"]) == 7
    # Four overlapping tiles, each 60% of the photo, covering the corners
    assert zero_shot.tiles(100, 50) == [(0, 0, 60, 30), (40, 0, 100, 30), (0, 20, 60, 50), (40, 20, 100, 50)]


def test_zero_shot_tiling_maps_tile_boxes_back_to_the_photo():
    # A fake model that always finds one box at the top-left of whatever it is given
    image = Image.new("RGB", (100, 50))
    fake = lambda img, phrases: (np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.5]))  # noqa: E731
    boxes, scores = zero_shot.detect_photo(fake, image, ["wound"], tiled=False)
    assert boxes.tolist() == [[0, 0, 10, 10]]
    # Tiled: the whole-photo box plus one per tile, shifted back to photo coordinates
    boxes, _ = zero_shot.detect_photo(fake, image, ["wound"], tiled=True)
    assert boxes.tolist() == [[0, 0, 10, 10], [0, 0, 10, 10], [40, 0, 50, 10], [0, 20, 10, 30], [40, 20, 50, 30]]


def test_zero_shot_run_writes_every_config_and_guards_test(world, tmp_path):
    # Fake detector whose score is the number of prompt phrases / 10, so configs are told apart
    photos = zero_shot.photo_table(world["box_manifest"], ("validation",))
    fake = lambda img, phrases: (np.array([[5.0, 5.0, 40.0, 30.0]]), np.array([len(phrases) / 10]))  # noqa: E731
    zero_shot.run(fake, "gdino", [c for c in zero_shot.config_order() if c.startswith("gdino_")], photos,
                  world["raw"], tmp_path / "det", log=lambda m: None)
    # One detections file per Grounding DINO config; P3 has 7 phrases so every score is 0.7
    files = sorted(p.stem for p in (tmp_path / "det").glob("*.csv"))
    assert len(files) == 8 and "gdino_P4_tiled" in files
    p3 = pd.read_csv(tmp_path / "det" / "gdino_P3.csv")
    assert len(p3) == len(photos) and p3["score"].eq(0.7).all()
    with pytest.raises(FileExistsError):  # detections files are written once
        zero_shot.run(fake, "gdino", ["gdino_P1"], photos, world["raw"], tmp_path / "det", log=lambda m: None)
    # Asking for the test split before a selection exists must be refused
    with pytest.raises(SystemExit, match="selection.json"):
        zero_shot.parse_args(["--model", "gdino", "--splits", "test", "--config", "gdino_P1",
                              "--output-dir", "x", "--image-root", "y"])


# ---------- 4P ----------

def test_public_configs_grey_input_and_source_classes(world, tmp_path):
    # Three YOLOv8 sizes, each on colour and greyscale input
    assert public.config_order() == ["v8n_colour", "v8n_grey", "v8m_colour", "v8m_grey", "v8x_colour", "v8x_grey"]
    # Grey input keeps three channels, all equal
    grey = np.asarray(public.as_input(Image.fromarray(np.uint8([[[255, 0, 0]]])), "grey"))
    assert grey.shape == (1, 1, 3) and len(set(grey[0, 0])) == 1
    # The public model's own class name is kept in a source_class column, one row per photo
    photos = public.photo_table(world["box_manifest"], ("validation",))
    fake = lambda img: (np.array([[1.0, 1.0, 30.0, 30.0]]), np.array([0.4]), ["gsw_entrance"])  # noqa: E731
    public.run({"v8n_colour": fake}, photos, world["raw"], tmp_path / "det", log=lambda m: None)
    written = pd.read_csv(tmp_path / "det" / "v8n_colour.csv")
    assert set(written["source_class"]) == {"gsw_entrance"} and len(written) == len(photos)


def test_willie_smoke_records_a_failed_load(tmp_path):
    # A missing repo and weights must be recorded as a failed load, not crash
    result = public.willie_smoke(tmp_path / "no_repo", tmp_path / "none.pt", tmp_path / "willie.json")
    assert result["loaded"] is False and "error" in json.loads((tmp_path / "willie.json").read_text())


# ---------- 4F ----------

def test_augment_flips_boxes_with_the_image():
    # A random source that always flips and never scales
    class Always:
        def random(self):
            return 0.0

        def uniform(self, a, b):
            return 1.0
    # Boxes are mirrored with the image both ways, and the image size is unchanged
    image, boxes = finetune.augment(Image.new("RGB", (100, 50)), np.array([[10.0, 5.0, 30.0, 15.0]]), Always())
    assert boxes.tolist() == [[70, 35, 90, 45]] and image.size == (100, 50)


def test_finetune_trains_detects_and_cross_fits(world, tmp_path, monkeypatch):
    # Shrink the detector input and warm-up so a CPU training step is quick
    monkeypatch.setattr(finetune, "MIN_SIZE", 64)
    monkeypatch.setattr(finetune, "MAX_SIZE", 96)
    monkeypatch.setattr(finetune, "WARMUP_ITERATIONS", 2)
    common = ["--image-root", str(world["raw"]), "--box-manifest", str(world["box_manifest"]),
              "--reference", str(world["reference_csv"]), "--limits", str(world["limits"]), "--device", "cpu"]
    # Train one epoch from scratch: checkpoint and history written
    run = tmp_path / "lr0.01"
    assert finetune.main(["train", "--lr", "0.01", "--output-dir", str(run), "--max-epochs", "1", "--no-pretrained", *common]) == 0
    assert (run / "best.pt").is_file() and len(pd.read_csv(run / "history.csv")) == 1
    # Validation detections only cover validation photos
    validation = pd.read_csv(run / "validation.csv")
    val_photos = set(world["reference"].loc[world["reference"]["split"] == "validation", "relative_path"])
    assert set(validation["relative_path"]) <= val_photos

    # Cross-fitting: train with fold 1 held out
    fold = tmp_path / "fold1"
    assert finetune.main(["train", "--lr", "0.01", "--output-dir", str(fold), "--holdout-fold", "1", "--epochs", "1",
                          "--cv-folds", str(world["cv_folds"]), "--no-pretrained", *common]) == 0
    # Detections are written only for fold-1 training photos, and the model trained on the other folds
    boxed = set(pd.read_csv(fold / "train_fold1.csv")["relative_path"])
    folds = pd.read_csv(world["cv_folds"], dtype=str).set_index("pseudonimized_id")["fold"]
    train = world["reference"][world["reference"]["split"] == "train"].drop_duplicates("relative_path")
    assert boxed <= set(train.loc[train["pseudonimized_id"].map(folds) == "1", "relative_path"])
    assert json.loads((fold / "run_config.json").read_text())["train_photos"] == 4  # 6 training photos minus fold 1

    # Detecting on test before a frozen selection must be refused
    with pytest.raises(SystemExit, match="selection.json"):
        finetune.main(["detect", "--run-dir", str(run), "--splits", "test", "--output", str(tmp_path / "t.csv"), *common])


# ---------- auto crops ----------

def frozen_selection(world, path, passed=True, threshold=0.5):
    # A frozen selection.json for arm 4F with the world's limits
    limits = json.loads(world["limits"].read_text())
    path.write_text(json.dumps({"arm": "4F", "chosen_config": "good", "threshold": threshold, "gate_passed": passed,
                                "limits": {k: limits[k] for k in ("max_box_share", "max_boxes_per_photo", "size_bins")}}))
    return path


def test_auto_crops_follow_the_final_boxes_and_fallbacks(world, tmp_path):
    # Good detections everywhere, except one test photo with none and one with only low scores
    ref = world["reference"]
    test_photos = sorted(set(ref.loc[ref["split"] == "test", "relative_path"]))
    dev = good_detections(ref, ("train", "validation"))
    test = good_detections(ref, ("test",), skip={test_photos[0]}, low={test_photos[1]})
    dev.to_csv(tmp_path / "dev.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    # Cut auto crops for both designs (union and per box) from the frozen selection
    sel = frozen_selection(world, tmp_path / "selection.json")
    common = ["--selection", str(sel), "--detections", str(tmp_path / "dev.csv"), str(tmp_path / "test.csv"),
              "--box-manifest", str(world["box_manifest"]), "--limits", str(world["limits"])]
    out = tmp_path / "roi_auto"
    assert crops_main([*common, "--output-dir", str(out), "--image-root", str(world["raw"]),
                       "--crop-root", str(tmp_path / "crops")]) == 0
    # Every photo gets a union crop; no detections means the full image, low scores means the top box
    union = pd.read_csv(out / "union" / "roi_manifest.csv").set_index("relative_path")
    box = pd.read_csv(out / "box" / "roi_manifest.csv")
    assert len(union) == 18 and union.loc[test_photos[0], "fallback"] == "full_image"
    assert union.loc[test_photos[1], "fallback"] == "top_box"
    assert tuple(union.loc[test_photos[0], ["x1", "y1", "x2", "y2"]]) == (0, 0, *SIZE)
    # A per-box crop is its detected box padded the T03 way; all splits present and the box cap respected
    first = box[box["fallback"] == "none"].iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == padded_crop_box([[first.box_x1, first.box_y1, first.box_x2, first.box_y2]], *SIZE)
    assert set(box["split"]) == {"train", "validation", "test"} and (box.groupby("relative_path").size() <= 35).all()
    # The saved crop image has exactly the crop box size
    crop = Image.open(tmp_path / "crops_box" / first.roi_relative_path)
    assert crop.size == (first.x2 - first.x1, first.y2 - first.y1)
    # The summary counts fallbacks per split
    summary = json.loads((out / "crop_summary.json").read_text())
    assert summary["fallback_by_split"]["test"] == {"full_image": 1, "top_box": 1, "none": 4}

    # The same detections file given twice must be refused
    with pytest.raises(SystemExit, match="more than once"):
        crops_main(["--selection", str(sel), "--detections", str(tmp_path / "dev.csv"), str(tmp_path / "dev.csv"),
                    "--box-manifest", str(world["box_manifest"]), "--limits", str(world["limits"]),
                    "--output-dir", str(tmp_path / "x")])
    # Two files that both cover the same photos must also be refused
    dev.to_csv(tmp_path / "dev_copy.csv", index=False)
    with pytest.raises(SystemExit, match="more than one"):
        crops_main(["--selection", str(sel), "--detections", str(tmp_path / "dev.csv"), str(tmp_path / "dev_copy.csv"),
                    "--box-manifest", str(world["box_manifest"]), "--limits", str(world["limits"]),
                    "--output-dir", str(tmp_path / "x")])
    # An arm that failed its recall gate gets no crops
    failed = frozen_selection(world, tmp_path / "failed.json", passed=False)
    with pytest.raises(SystemExit, match="gate"):
        crops_main(["--selection", str(failed), *common[2:], "--output-dir", str(tmp_path / "y")])


# ---------- scenario ids ----------

def test_arm_scenario_ids_are_reserved_by_arm():
    # Each arm owns its own block of scenario ids, 24 per arm (3 designs x 8 families)
    assert arm_scenario_ids("4Z")[("roi", "efficientnet_b0")] == "S33"
    assert arm_scenario_ids("4P")[("box", "efficientnet_b0")] == "S65"
    assert arm_scenario_ids("4F")[("crop", "alexnet_ensemble")] == "S104"
    assert len(set(arm_scenario_ids("4F").values())) == 24


# ---------- existing models on other crops ----------

def test_existing_models_reproduce_their_reported_predictions(tmp_path):
    from src.evaluation.predict_on_manifest import main as predict_main
    from src.training.cnn_svm import main as svm_main
    from src.training.train import main as train_main
    from tests.synthetic_data import build_synthetic_dataset
    from tests.test_cnn_svm import svm_args
    from tests.test_training import run_args

    # Train a small CNN and a CNN-SVM on synthetic ROI crops as a stand-in main run
    data = build_synthetic_dataset(tmp_path / "data")
    main_root = tmp_path / "main"
    assert train_main(run_args("mobilenet_v3_small", "roi", data, main_root / "roi" / "mobilenet_v3_small", "--smoke")) == 0
    assert svm_main(svm_args(data, main_root / "roi" / "cnn_svm")) == 0
    # Re-run them on the manual union crops with --check-manual
    out = tmp_path / "primary"
    assert predict_main(["--experiments-root", str(main_root), "--design", "union", "--manifest", str(data["roi_manifest"]),
                         "--image-root", str(data["roi_root"]), "--output-root", str(out), "--device", "cpu",
                         "--num-workers", "0", "--families", "mobilenet_v3_small", "cnn_svm", "--check-manual"]) == 0
    # On the same crops they must reproduce the reported predictions
    for check in json.loads((out / "check_manual_union.json").read_text()):
        assert check["same_predicted_class"] == 1.0 and check["max_abs_probability_difference"] < 1e-5, check
    # Predictions are written for the test split only
    written = pd.read_csv(out / "roi" / "mobilenet_v3_small" / "test_predictions.csv")
    assert set(written["split"]) == {"test"}


# ---------- comparisons and verification ----------

FAMILIES = ("efficientnet_b0", "resnet152", "mobilenet_v3_small", "googlenet", "vgg16", "vit_b16", "cnn_svm", "alexnet_ensemble")


def write_predictions(path, rows, probs, photo=None):
    # A prediction file in the training format for three classes, with an optional photo column
    classes = ["a", "b", "c"]
    frame = pd.DataFrame({"relative_path": [r[0] for r in rows], **({"source_relative_path": photo} if photo is not None else {}),
                          "pseudonimized_id": [r[1] for r in rows], "split": "test", "true_class": [r[2] for r in rows],
                          "predicted_class": [classes[i] for i in probs.argmax(1)]})
    for j, c in enumerate(classes):
        frame[f"prob_{c}"] = probs[:, j]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture()
def comparison_world(tmp_path):
    # 30 test photos in 15 cases, with union-crop rows and two box-crop rows per photo
    rng = np.random.default_rng(3)
    exp, data = tmp_path / "exp", tmp_path / "data"
    photos = [(f"{c}/P{i}.JPG", str(i // 2), "abc"[i % 3]) for i in range(30) for c in ["x"]]
    union_rows = [(f"test/{lab}/P{i}_roi.jpg", case, lab) for i, (_, case, lab) in enumerate(photos)]
    crop_rows = [(f"test/{lab}/P{i}_bbox0{k}.jpg", case, lab) for i, (_, case, lab) in enumerate(photos) for k in (1, 2)]
    crop_photos = [photos[i][0] for i in range(len(photos)) for _ in (1, 2)]

    def probs(n, labels):
        # Random probabilities, boosted on the true class for about 70% of rows
        p = rng.dirichlet(np.ones(3), size=n)
        hit = rng.random(n) < 0.7
        p[hit, ["abc".index(l) for l in np.array(labels)[hit]]] += 2
        return p / p.sum(1, keepdims=True)

    # Per family: auto-crop predictions identical to the manual ones, for both union and box designs
    for f in FAMILIES:
        write_predictions(exp / "main" / "full" / f / "test_predictions.csv", union_rows, probs(30, [r[2] for r in union_rows]))
        manual_union = probs(30, [r[2] for r in union_rows])
        write_predictions(exp / "main" / "roi" / f / "test_predictions.csv", union_rows, manual_union)
        write_predictions(exp / "auto_4F" / "primary" / "roi" / f / "test_predictions.csv", union_rows, manual_union)
        manual_box = probs(60, [r[2] for r in crop_rows])
        write_predictions(exp / "roi_box" / "box" / f / "test_predictions.csv", crop_rows, manual_box, crop_photos)
        write_predictions(exp / "auto_4F" / "primary" / "box" / f / "test_predictions.csv", crop_rows, manual_box, crop_photos)
    # 4F passed its gate, 4Z failed
    for arm, passed in (("4F", True), ("4Z", False)):
        (exp / f"auto_{arm}" / "detection").mkdir(parents=True, exist_ok=True)
        (exp / f"auto_{arm}" / "detection" / "selection.json").write_text(json.dumps(
            {"chosen_config": "c", "threshold": 0.5, "gate_passed": passed, "threshold_detail": {"recall": 0.8 if passed else 0.2}}))
    # 4F auto-crop manifests: one union fallback photo, and box crops of which half touch no manual box
    union = pd.DataFrame({"relative_path": [p for p, _, _ in photos], "roi_relative_path": [r[0] for r in union_rows],
                          "split": "test", "fallback": ["full_image"] + ["none"] * 29})
    (data / "4F" / "union").mkdir(parents=True)
    (data / "4F" / "box").mkdir(parents=True)
    union.to_csv(data / "4F" / "union" / "roi_manifest.csv", index=False)
    pd.DataFrame({"relative_path": crop_photos, "roi_relative_path": [r[0] for r in crop_rows], "split": "test",
                  "fallback": "none", "box_x1": [0, 50] * 30, "box_y1": 0, "box_x2": [10, 60] * 30, "box_y2": 10}
                 ).to_csv(data / "4F" / "box" / "roi_manifest.csv", index=False)
    # Reference boxes and the manual union manifest the comparison reads
    pd.DataFrame({"relative_path": [p for p, _, _ in photos], "split": "test", "x1": 0, "y1": 0, "x2": 10, "y2": 10}
                 ).to_csv(data / "reference_boxes.csv", index=False)
    union[["relative_path", "roi_relative_path"]].to_csv(data / "manual_union.csv", index=False)
    return exp, data


def test_identical_predictions_are_not_inferior_and_failed_arms_are_skipped(comparison_world, tmp_path):
    from src.evaluation.auto_vs_manual import main as compare_main, verdict
    from src.evaluation.verify import Report
    from src.evaluation.verify_auto import check_numbers, check_primary

    # Verdict from the CI against a 0.03 margin: above it, wholly below it, or straddling it
    assert verdict(-0.02, 0.05, 0.03) == "not inferior" and verdict(-0.10, -0.04, 0.03) == "inferior"
    assert verdict(-0.05, 0.01, 0.03) == "inconclusive"
    # Compare both arms against manual crops
    exp, data = comparison_world
    out = exp / "auto_summary"
    assert compare_main(["--experiments-dir", str(exp), "--arms", "4Z", "4F", "--manifest-root", str(data),
                         "--union-manifest", str(data / "manual_union.csv"), "--output-dir", str(out), "--bootstrap", "30"]) == 0
    # Only the passed arm 4F is compared; identical predictions give difference 0 and 'not inferior'
    primary = pd.read_csv(out / "primary_noninferiority.csv")
    assert list(primary["arm"]) == ["4F", "4F"] and list(primary["design"]) == ["union", "box"]
    assert (primary["difference"] == 0).all() and (primary["verdict"] == "not inferior").all()
    assert (primary["p_noninferiority"] == 0).all() and primary.loc[0, "fallback_photos"] == 1
    # The failed arm is reported as such; half of 4F's test boxes touch no manual box
    arms = json.loads((out / "arms.json").read_text())
    assert arms["4Z"]["gate_passed"] is False and arms["4F"]["test_auto_boxes_touching_no_manual_box"] == pytest.approx(0.5)
    assert len(pd.read_csv(out / "primary_by_family.csv")) == 2 * 3 * 9  # arm-designs x metrics x (8 families + pooled)

    # Verification recomputes the numbers and the primary comparison and finds nothing wrong
    report = Report()
    check_numbers(report, exp, ["4F"])
    check_primary(report, "4F", exp / "auto_4F", data)
    assert report.rows and not report.failed, report.failed


def test_verification_of_the_real_reference_boxes():
    from src.evaluation.verify import Report
    from src.evaluation.verify_auto import check_reference

    # The committed reference boxes must agree with the committed per-box manifest
    report = Report()
    check_reference(report, Path("data/roi_auto"), Path("data/roi_box/roi_manifest.csv"))
    assert [r["status"] for r in report.rows] == ["PASS", "PASS"], report.rows


def test_networks_are_rebuilt_with_their_runs_pretrained_setting(tmp_path, monkeypatch):
    """GoogLeNet's transform_input depends on pretrained; rebuilding without it changed 41% of predictions."""
    import torch
    from src.evaluation import predict_on_manifest as pm

    seen = []
    monkeypatch.setattr(pm, "build_model", lambda family, n, pretrained: seen.append(pretrained) or torch.nn.Linear(1, n))
    monkeypatch.setattr(pm, "load_checkpoint", lambda path, device: {"model": torch.nn.Linear(1, 7).state_dict()})
    monkeypatch.setattr(pm, "loader_for", lambda *a, **k: None)
    monkeypatch.setattr(pm, "predict", lambda *a, **k: (0.0, None, np.zeros((0, 7))))
    for pretrained in (True, False):
        (tmp_path / "run_config.json").write_text(json.dumps({"classes": list("abcdefg"), "image_size": 64,
                                                               "batch_size": 2, "pretrained": pretrained}))
        pm.network_probs(tmp_path, "googlenet", None, torch.device("cpu"), 0)
    assert seen == [True, False]
