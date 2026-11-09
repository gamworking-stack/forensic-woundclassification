"""Tests for the realism and memorisation check of generated crops (T34): CPU, a stand-in detector."""

import pickle
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.data.export_generator_crops import main as export_main
from src.evaluation.generated_quality import kid, main as quality_main, nearest
from tests.synthetic_data import build_synthetic_box_dataset, build_synthetic_dataset


class TinyDetector(nn.Module):
    """Stand-in for StyleGAN's Inception: a 4 x 4 colour thumbnail of the uint8 image."""

    def forward(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(x.float() / 255, 4).flatten(1)


def test_kid_is_near_zero_for_one_distribution_and_grows_with_a_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 16)), rng.normal(size=(200, 16))
    assert abs(kid(a, b, rng, subsets=20)) < 0.05
    assert kid(a, b + 2, rng, subsets=20) > 1


def test_nearest_finds_an_exact_copy():
    reference = np.random.default_rng(1).normal(size=(10, 8))
    index, sim = nearest(reference[[3, 7]], reference)
    assert list(index) == [3, 7] and np.allclose(sim, 1)


def test_copies_of_training_crops_are_flagged(tmp_path):
    root = tmp_path / "wounds"
    boxes = build_synthetic_box_dataset(root, build_synthetic_dataset(root))
    train_dir = tmp_path / "train"
    export_main(["--manifest", str(boxes["box_manifest"]), "--image-root", str(boxes["box_root"]),
                 "--output-dir", str(train_dir), "--size", "32"])
    # "Generated" crops that are exact copies of the training scratch crops
    crops = pd.read_csv(train_dir / "crops.csv")
    scratch = crops[crops["class"] == "scratch"]
    samples = tmp_path / "samples"
    (samples / "scratch").mkdir(parents=True)
    rows = []
    for i, file in enumerate(scratch["file"]):
        shutil.copy(train_dir / file, samples / "scratch" / f"copy_{i}.png")
        rows.append({"file": f"scratch/copy_{i}.png", "class": "scratch", "generator": "copier",
                     "checkpoint": "-", "seed": i})
    pd.DataFrame(rows).to_csv(samples / "samples.csv", index=False)
    detector = tmp_path / "detector.pkl"
    with open(detector, "wb") as f:
        pickle.dump(TinyDetector(), f)

    out = tmp_path / "quality"
    quality_main(["--samples", str(samples), "--train-dir", str(train_dir), "--manifest", str(boxes["box_manifest"]),
                  "--image-root", str(boxes["box_root"]), "--detector", str(detector), "--output-dir", str(out),
                  "--size", "32"])
    quality = pd.read_csv(out / "quality.csv").iloc[0]
    assert quality["generated"] == len(scratch) and quality["flagged_share"] == 1
    near = pd.read_csv(out / "nearest.csv")
    assert sorted(near["nearest_train"]) == sorted(scratch["file"])
    assert (out / "panels" / "samples_copier_scratch_closest.png").is_file()


def test_refuses_output_inside_repo(tmp_path):
    from src.data.export_generator_crops import REPO_ROOT
    with pytest.raises(ValueError, match="inside the repository"):
        quality_main(["--samples", str(tmp_path), "--train-dir", str(tmp_path), "--image-root", str(tmp_path),
                      "--detector", str(tmp_path / "d.pkl"), "--output-dir", str(REPO_ROOT / "quality")])
