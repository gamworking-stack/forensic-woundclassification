"""T34 stage 2: generated crops join the train split only (train.py, cnn_svm.py) and how they are chosen."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.data.export_generator_crops import main as export_main
from src.data.select_generated import main as select_main
from src.training.cnn_svm import main as svm_main
from src.training.train import main as train_main
from tests.synthetic_data import build_synthetic_box_dataset, build_synthetic_dataset


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("wounds")
    base = build_synthetic_dataset(root)
    data = {**base, **build_synthetic_box_dataset(root, base)}
    # Three "generated" scratch crops outside the crop folder (copies of one training crop)
    manifest = pd.read_csv(data["box_manifest"], dtype=str)
    source = data["box_root"] / manifest.query("split == 'train' and canonical_wound_label == 'scratch'")["roi_relative_path"].iloc[0]
    generated = root / "generated"
    generated.mkdir()
    for i in range(3):
        shutil.copy(source, generated / f"scratch_{i}.png")
    extra = root / "extra_train.csv"
    pd.DataFrame({"path": [str(generated / f"scratch_{i}.png") for i in range(3)], "class": "scratch"}).to_csv(extra, index=False)
    return {**data, "extra": extra, "manifest": manifest}


def crop_args(data, output_dir, *extra):
    return ["--input-path", "crop", "--manifest", str(data["box_manifest"]), "--image-root", str(data["box_root"]),
            "--output-dir", str(output_dir), "--no-pretrained", "--device", "cpu", "--num-workers", "0",
            "--image-size", "64", "--extra-train", str(data["extra"]), *extra]


def check_real_predictions(data, out):
    # Generated crops are trained on but never predicted: prediction files hold the real crops only
    config = json.loads((out / "run_config.json").read_text())
    real_train = int((data["manifest"]["split"] == "train").sum())
    assert config["images_per_split"]["train"] == real_train + 3
    assert config["extra_train_per_class"] == {"scratch": 3}
    for split in ("validation", "test"):
        preds = pd.read_csv(out / f"{split}_predictions.csv", dtype=str)
        assert sorted(preds["relative_path"]) == sorted(data["manifest"].query("split == @split")["roi_relative_path"])


def test_train_adds_generated_crops_to_train_only(data, tmp_path):
    assert train_main(["--family", "mobilenet_v3_small", *crop_args(data, tmp_path, "--max-epochs", "1")]) == 0
    check_real_predictions(data, tmp_path)


def test_cnn_svm_adds_generated_crops_to_train_only(data, tmp_path):
    assert svm_main(crop_args(data, tmp_path)) == 0
    check_real_predictions(data, tmp_path)


def test_unknown_class_is_refused(data, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"path": [pd.read_csv(data["extra"])["path"][0]], "class": ["bite"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="bite"):
        train_main(["--family", "mobilenet_v3_small", *crop_args(data, tmp_path)[:-2], "--extra-train", str(bad)])


def test_select_tops_classes_up_with_unflagged_crops(data, tmp_path):
    train_dir = tmp_path / "train"
    export_main(["--manifest", str(data["box_manifest"]), "--image-root", str(data["box_root"]),
                 "--output-dir", str(train_dir), "--size", "32"])
    counts = pd.read_csv(train_dir / "crops.csv")["class"].value_counts()
    samples = tmp_path / "samples"
    (samples / "scratch").mkdir(parents=True)
    rows = []
    for i in range(10):
        shutil.copy(pd.read_csv(data["extra"])["path"][0], samples / "scratch" / f"g_{i}.png")
        rows.append({"file": f"scratch/g_{i}.png", "class": "scratch", "generator": "g", "seed": i})
    pd.DataFrame(rows).to_csv(samples / "samples.csv", index=False)
    # The first 5 are flagged as near-copies and must not be chosen
    pd.DataFrame({"file": [str(samples / r["file"]) for r in rows], "flagged": [True] * 5 + [False] * 5}).to_csv(
        tmp_path / "nearest.csv", index=False)
    target = int(counts["scratch"]) + 4
    select_main(["--samples", str(samples), "--nearest", str(tmp_path / "nearest.csv"), "--train-dir", str(train_dir),
                 "--classes", "scratch", "--target", str(target), "--output", str(tmp_path / "extra.csv")])
    chosen = pd.read_csv(tmp_path / "extra.csv")
    assert len(chosen) == 4 and set(chosen["seed"]) <= {5, 6, 7, 8, 9}
    with pytest.raises(ValueError, match="only 5 unflagged"):
        select_main(["--samples", str(samples), "--nearest", str(tmp_path / "nearest.csv"), "--train-dir", str(train_dir),
                     "--classes", "scratch", "--target", str(int(counts["scratch"]) + 6), "--output", str(tmp_path / "x.csv")])
