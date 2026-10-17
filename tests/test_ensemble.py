"""Tests for the soft-voting ensemble (hand-made members and a small end-to-end run, CPU)."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.ensemble import main as ensemble_main
from src.training.train import main as train_main
from tests.synthetic_data import EXPECTED_PREDICTION_COLUMNS, IMAGES_PER_CLASS, RAW_FOLDERS, build_synthetic_dataset


CLASSES = sorted(RAW_FOLDERS)
PROB_COLUMNS = [f"prob_{c.replace(' ', '_')}" for c in CLASSES]


def make_member(directory: Path, seed: int, probs: np.ndarray, f1: float, epoch: int = 5,
                order=None, classes=CLASSES, input_path="roi") -> Path:
    """Write a fake finished train.py run: run_config.json, best.pt and both prediction files."""
    directory.mkdir(parents=True)
    # Shared row identity: one image per case, true classes cycling through the seven wound types
    n = len(probs)
    rows = pd.DataFrame({
        "relative_path": [f"img_{i}.jpg" for i in range(n)],
        "pseudonimized_id": [str(i) for i in range(n)],
        "true_class": [CLASSES[i % len(CLASSES)] for i in range(n)],
    })
    # Write both prediction files with the given probabilities and their argmax as the predicted class
    for split in ("validation", "test"):
        frame = rows.copy()
        frame.insert(2, "split", split)
        frame["predicted_class"] = [classes[i] for i in probs.argmax(axis=1)]
        for j, name in enumerate(classes):
            frame[f"prob_{name.replace(' ', '_')}"] = probs[:, j]
        # Optionally shuffle the rows to simulate a member whose files are in a different order
        if order is not None:
            frame = frame.iloc[order].reset_index(drop=True)
        frame.to_csv(directory / f"{split}_predictions.csv", index=False)
    # Config and checkpoint carry the seed and validation F1 the ensemble reads to pick its representative
    config = {"family": "alexnet", "input_path": input_path, "seed": seed, "classes": list(classes)}
    (directory / "run_config.json").write_text(json.dumps(config))
    torch.save({"epoch": epoch, "validation_macro_f1": f1}, directory / "best.pt")
    return directory


def random_probs(rng, n=6):
    # Random rows normalised to sum to 1, like softmax outputs
    raw = rng.random((n, len(CLASSES)))
    return raw / raw.sum(axis=1, keepdims=True)


def test_soft_vote_is_the_mean_of_member_probabilities(tmp_path):
    # Three members with different probabilities; the last has the highest validation F1
    rng = np.random.default_rng(0)
    probs = [random_probs(rng) for _ in range(3)]
    members = [make_member(tmp_path / f"member_0{k + 1}", 43 + k, probs[k], f1=0.5 + 0.1 * k)
               for k in range(3)]
    out = tmp_path / "ensemble"
    assert ensemble_main(["--members", *map(str, members), "--output-dir", str(out)]) == 0

    # Soft vote = plain mean of member probabilities, and the predicted class is its argmax
    expected = np.mean(probs, axis=0)
    for split in ("validation", "test"):
        voted = pd.read_csv(out / f"{split}_predictions.csv")
        assert voted.columns.tolist() == EXPECTED_PREDICTION_COLUMNS
        assert np.allclose(voted[PROB_COLUMNS].to_numpy(), expected)
        assert voted["predicted_class"].tolist() == [CLASSES[i] for i in expected.argmax(axis=1)]

    # Member table lists seeds in order and marks the best-F1 member as the representative
    table = pd.read_csv(out / "ensemble_members.csv")
    assert table["seed"].tolist() == [43, 44, 45]
    assert table["representative"].tolist() == [False, False, True]
    assert json.loads((out / "run_config.json").read_text())["combination"] == "mean"


def test_representative_tie_goes_to_lower_seed(tmp_path):
    # All members tie on F1 and are given out of seed order
    rng = np.random.default_rng(1)
    members = [make_member(tmp_path / f"m{seed}", seed, random_probs(rng), f1=0.7) for seed in (45, 43, 44)]
    out = tmp_path / "ensemble"
    assert ensemble_main(["--members", *map(str, members), "--output-dir", str(out)]) == 0
    # The tie must go to the lowest seed, not to the first member listed
    table = pd.read_csv(out / "ensemble_members.csv")
    assert table.loc[table["representative"], "seed"].tolist() == [43]


def test_rows_in_different_order_are_an_error(tmp_path):
    # Member b has its rows shuffled, so averaging row by row would mix up images
    rng = np.random.default_rng(2)
    a = make_member(tmp_path / "a", 43, random_probs(rng), 0.5)
    b = make_member(tmp_path / "b", 44, random_probs(rng), 0.5, order=[1, 0, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="rows differ"):
        ensemble_main(["--members", str(a), str(b), "--output-dir", str(tmp_path / "out")])


def test_missing_prediction_file_is_an_error(tmp_path):
    rng = np.random.default_rng(3)
    a = make_member(tmp_path / "a", 43, random_probs(rng), 0.5)
    b = make_member(tmp_path / "b", 44, random_probs(rng), 0.5)
    # Delete one member's test file; the ensemble must refuse rather than vote with two members
    (b / "test_predictions.csv").unlink()
    with pytest.raises(FileNotFoundError, match="test_predictions.csv"):
        ensemble_main(["--members", str(a), str(b), "--output-dir", str(tmp_path / "out")])


def test_mismatched_members_are_an_error(tmp_path):
    # Members whose class columns are in a different order cannot be averaged
    rng = np.random.default_rng(4)
    a = make_member(tmp_path / "a", 43, random_probs(rng), 0.5)
    other_classes = CLASSES[1:] + CLASSES[:1]
    b = make_member(tmp_path / "b", 44, random_probs(rng), 0.5, classes=other_classes)
    with pytest.raises(ValueError, match="classes"):
        ensemble_main(["--members", str(a), str(b), "--output-dir", str(tmp_path / "out1")])

    # Two members with the same seed are not independent runs
    c = make_member(tmp_path / "c", 43, random_probs(rng), 0.5)
    with pytest.raises(ValueError, match="distinct seeds"):
        ensemble_main(["--members", str(a), str(c), "--output-dir", str(tmp_path / "out2")])

    # Members trained on different inputs (ROI vs full photo) cannot be combined
    d = make_member(tmp_path / "d", 45, random_probs(rng), 0.5, input_path="full")
    with pytest.raises(ValueError, match="input_path"):
        ensemble_main(["--members", str(a), str(d), "--output-dir", str(tmp_path / "out3")])


def test_end_to_end_with_three_alexnet_members(tmp_path):
    # Train three real AlexNet smoke runs on synthetic ROI crops, one per seed
    data = build_synthetic_dataset(tmp_path / "data")
    members = []
    for k, seed in enumerate((43, 44, 45), start=1):
        member = tmp_path / "alexnet_ensemble" / f"member_0{k}"
        assert train_main([
            "--family", "alexnet", "--input-path", "roi",
            "--manifest", str(data["roi_manifest"]), "--image-root", str(data["roi_root"]),
            "--output-dir", str(member), "--seed", str(seed),
            "--image-size", "64", "--no-pretrained", "--device", "cpu", "--num-workers", "0", "--smoke",
        ]) == 0
        members.append(member)

    # Combine them and check the voted files have the training format and valid probabilities
    out = tmp_path / "alexnet_ensemble"
    assert ensemble_main(["--members", *map(str, members), "--output-dir", str(out)]) == 0
    for split in ("validation", "test"):
        voted = pd.read_csv(out / f"{split}_predictions.csv")
        assert voted.columns.tolist() == EXPECTED_PREDICTION_COLUMNS
        assert len(voted) == IMAGES_PER_CLASS * len(RAW_FOLDERS)
        assert np.allclose(voted[PROB_COLUMNS].sum(axis=1), 1.0, atol=1e-4)
    # Exactly one member is marked as the representative
    assert pd.read_csv(out / "ensemble_members.csv")["representative"].sum() == 1
