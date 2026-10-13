"""T14 Grad-CAM: maps for every family, the fixed example draw and an end-to-end run (CPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.gradcam import draw_examples, grad_cam, main as gradcam_main, mass_in_box, wound_box
from src.training.config import FAMILIES
from src.training.models import build_model
from src.training.train import main as train_main
from tests.synthetic_data import RAW_FOLDERS, build_synthetic_dataset
from tests.test_training import run_args


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return build_synthetic_dataset(tmp_path_factory.mktemp("wounds"))


@pytest.mark.parametrize("family", FAMILIES)
def test_map_matches_the_input_for_every_family(family):
    torch.manual_seed(0)
    model = build_model(family, 7, pretrained=False).eval()
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    size = 224 if family == "vit_b16" else 64
    cam, probs = grad_cam(model, family, torch.randn(1, 3, size, size))
    assert cam.shape == (size, size)
    assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
    assert probs.shape == (7,) and probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_example_draw_is_fixed_and_balanced(data):
    a = draw_examples(data["roi_manifest"], 2, 42)
    b = draw_examples(data["roi_manifest"], 2, 42)
    pd.testing.assert_frame_equal(a, b)
    assert (a["split"] == "test").all()
    assert a.groupby("canonical_wound_label").size().eq(2).all() and len(a) == 2 * len(RAW_FOLDERS)
    assert not a["pseudonimized_id"].duplicated().any()
    assert sorted(a["set"].unique()) == [1, 2]


def test_wound_box_maps_into_the_padded_input():
    row = {"x1": 100, "y1": 0, "x2": 300, "y2": 200, "source_width": 400, "source_height": 200}
    assert wound_box(row, 100) == pytest.approx((25.0, 25.0, 75.0, 75.0))  # 400x400 square, 50 px offset in y
    cam = np.zeros((100, 100)); cam[25:75, 25:75] = 1.0
    assert mass_in_box(cam, wound_box(row, 100)) == pytest.approx((1.0, 0.25))
    assert np.isnan(mass_in_box(cam, None)[0])


def test_end_to_end_on_synthetic_runs(data, tmp_path):
    exp = tmp_path / "exp"
    for path in ("full", "roi"):
        assert train_main(run_args("mobilenet_v3_small", path, data, exp / path / "mobilenet_v3_small", "--smoke")) == 0
        member = exp / path / "alexnet_ensemble" / "member_01"
        assert train_main(run_args("alexnet", path, data, member, "--smoke", "--seed", "43")) == 0
        pd.DataFrame([{"member": 1, "seed": 43, "best_epoch": 1, "best_validation_macro_f1": 0.5,
                       "output_dir": str(member), "representative": True}]) \
            .to_csv(exp / path / "alexnet_ensemble" / "ensemble_members.csv", index=False)

    args = ["--experiments-root", str(exp), "--roi-manifest", str(data["roi_manifest"]),
            "--families", "mobilenet_v3_small", "alexnet_ensemble", "--no-pretrained", "--device", "cpu"]
    assert gradcam_main(args) == 0
    results = pd.read_csv(exp / "gradcam" / "gradcam_results.csv")
    assert len(results) == 2 * 2 * 14
    assert results["matches_prediction_file"].all()
    assert results.loc[results["family"] == "alexnet_ensemble", "checkpoint"].str.contains("member_01").all()
    for family in ("mobilenet_v3_small", "alexnet_ensemble"):
        for k in (1, 2):
            assert (exp / "gradcam" / f"{family}_set{k}.png").is_file()
