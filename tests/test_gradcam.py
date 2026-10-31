"""T14 Grad-CAM: maps for every family, the fixed example draw, an end-to-end run and the panels (CPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.gradcam import all_test_images, draw_examples, grad_cam, main as gradcam_main, mass_in_box, wound_box
from src.evaluation.gradcam_panels import main as panels_main
from src.training.config import FAMILIES
from src.training.models import build_model
from src.training.train import main as train_main
from tests.synthetic_data import RAW_FOLDERS, build_synthetic_box_dataset, build_synthetic_dataset
from tests.test_training import run_args


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return build_synthetic_dataset(tmp_path_factory.mktemp("wounds"))


@pytest.mark.parametrize("family", FAMILIES)
def test_map_matches_the_input_for_every_family(family):
    torch.manual_seed(0)
    # Untrained 7-class model of this family, seeded so the run is repeatable
    model = build_model(family, 7, pretrained=False).eval()
    # Turn off in-place activations, which would break the gradient hooks
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    # ViT needs its native 224 px input; small images are enough for the CNNs
    size = 224 if family == "vit_b16" else 64
    cam, probs = grad_cam(model, family, torch.randn(1, 3, size, size))
    # Map has the input size, values in [0, 1], and the probabilities form a distribution
    assert cam.shape == (size, size)
    assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
    assert probs.shape == (7,) and probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_example_draw_is_fixed_and_balanced(data):
    # Same seed twice must pick the same example images
    a = draw_examples(data["roi_manifest"], 2, 42)
    b = draw_examples(data["roi_manifest"], 2, 42)
    pd.testing.assert_frame_equal(a, b)
    # Only test images, 2 per class, from distinct cases, numbered as sets 1 and 2
    assert (a["split"] == "test").all()
    assert a.groupby("canonical_wound_label").size().eq(2).all() and len(a) == 2 * len(RAW_FOLDERS)
    assert not a["pseudonimized_id"].duplicated().any()
    assert sorted(a["set"].unique()) == [1, 2]


def test_examples_never_reuse_a_case_across_classes(tmp_path):
    # Case "1" has a test image of both classes; each class has one more case of its own
    rows = [("1", "cut wound", "a"), ("1", "scratch", "b"), ("2", "cut wound", "c"), ("3", "scratch", "d")]
    manifest = tmp_path / "roi_manifest.csv"
    pd.DataFrame([{"pseudonimized_id": c, "canonical_wound_label": l, "roi_relative_path": f"test/{l}/{p}.jpg",
                   "split": "test"} for c, l, p in rows]).to_csv(manifest, index=False)
    # Whatever the seed, the two picks come from two different cases
    for seed in range(20):
        examples = draw_examples(manifest, 1, seed)
        assert not examples["pseudonimized_id"].duplicated().any(), seed


def test_all_test_images_takes_every_test_image_of_the_classes(data):
    # Selected rows must be exactly the manifest's test images of the two classes
    rows = all_test_images(data["roi_manifest"], ["cut wound", "scratch"])
    manifest = pd.read_csv(data["roi_manifest"], dtype=str)
    expected = manifest[(manifest["split"] == "test") & manifest["canonical_wound_label"].isin(["cut wound", "scratch"])]
    assert sorted(rows["roi_relative_path"]) == sorted(expected["roi_relative_path"])
    # Within each class the images are numbered 1, 2, 3, ... with no gaps
    assert rows.groupby("canonical_wound_label")["set"].apply(list).map(lambda s: s == list(range(1, len(s) + 1))).all()
    # An unknown class name must fail rather than return nothing
    with pytest.raises(ValueError, match="No test images"):
        all_test_images(data["roi_manifest"], ["cut"])


def test_wound_box_maps_into_the_padded_input():
    # Wide 400x200 photo with a box in the middle, padded to a square then resized to 100 px
    row = {"x1": 100, "y1": 0, "x2": 300, "y2": 200, "source_width": 400, "source_height": 200}
    assert wound_box(row, 100) == pytest.approx((25.0, 25.0, 75.0, 75.0))  # 400x400 square, 50 px offset in y
    # A map lit only inside the box puts all its mass there, over a quarter of the area
    cam = np.zeros((100, 100)); cam[25:75, 25:75] = 1.0
    assert mass_in_box(cam, wound_box(row, 100)) == pytest.approx((1.0, 0.25))
    # No box means the share is undefined, not zero
    assert np.isnan(mass_in_box(cam, None)[0])


def test_end_to_end_on_synthetic_runs(data, tmp_path):
    # Train one smoke model and one single-member AlexNet ensemble for both full and ROI inputs
    exp = tmp_path / "exp"
    for path in ("full", "roi"):
        assert train_main(run_args("mobilenet_v3_small", path, data, exp / path / "mobilenet_v3_small", "--smoke")) == 0
        member = exp / path / "alexnet_ensemble" / "member_01"
        assert train_main(run_args("alexnet", path, data, member, "--smoke", "--seed", "43")) == 0
        # Ensemble members file that marks the member as representative
        pd.DataFrame([{"member": 1, "seed": 43, "best_epoch": 1, "best_validation_macro_f1": 0.5,
                       "output_dir": str(member), "representative": True}]) \
            .to_csv(exp / path / "alexnet_ensemble" / "ensemble_members.csv", index=False)

    # Run Grad-CAM on CPU for both families
    args = ["--experiments-root", str(exp), "--roi-manifest", str(data["roi_manifest"]),
            "--families", "mobilenet_v3_small", "alexnet_ensemble", "--no-pretrained", "--device", "cpu"]
    assert gradcam_main(args) == 0
    # 2 families x 2 inputs x 14 examples; predictions agree with the saved prediction files
    results = pd.read_csv(exp / "gradcam" / "gradcam_results.csv")
    assert len(results) == 2 * 2 * 14
    assert results["matches_prediction_file"].all()
    # The ensemble is explained through its representative member's checkpoint
    assert results.loc[results["family"] == "alexnet_ensemble", "checkpoint"].str.contains("member_01").all()
    # One figure per family and example set
    for family in ("mobilenet_v3_small", "alexnet_ensemble"):
        for k in (1, 2):
            assert (exp / "gradcam" / f"{family}_set{k}.png").is_file()


def test_panels_explain_every_test_photo_and_crop(data, tmp_path):
    # Per-box crops of the synthetic photos, a full-image model (main run) and a per-crop model
    box = build_synthetic_box_dataset(tmp_path, data)
    exp = tmp_path / "exp"
    assert train_main(run_args("mobilenet_v3_small", "full", data, exp / "full" / "mobilenet_v3_small", "--smoke")) == 0
    assert train_main(["--family", "mobilenet_v3_small", "--input-path", "crop", "--manifest", str(box["box_manifest"]),
                       "--image-root", str(box["box_root"]), "--output-dir", str(exp / "crop" / "mobilenet_v3_small"),
                       "--no-pretrained", "--device", "cpu", "--num-workers", "0", "--image-size", "64",
                       "--max-epochs", "1"]) == 0

    # Panels for every test photo and crop
    out = tmp_path / "panels"
    assert panels_main(["--main-root", str(exp), "--crop-root", str(exp), "--box-manifest", str(box["box_manifest"]),
                        "--all-test", "--families", "mobilenet_v3_small", "--output-dir", str(out),
                        "--device", "cpu", "--no-pretrained"]) == 0
    test = pd.read_csv(box["box_manifest"], dtype=str)
    test = test[test["split"] == "test"]
    panels = pd.read_csv(out / "panels.csv", dtype=str, keep_default_na=False)
    # One full-image panel per test photo and one per crop, each agreeing with its prediction file
    assert sorted(panels.loc[panels["input_path"] == "full", "photo"]) == sorted(test["relative_path"].unique())
    assert sorted(panels.loc[panels["input_path"] == "crop", "crop"]) == sorted(test["roi_relative_path"])
    assert (panels["matches_prediction_file"] == "True").all()
    # A file per panel in its class folder, and one overview page per class
    family = out / "mobilenet_v3_small"
    assert len(list(family.glob("*/*.png"))) == len(panels)
    assert {p.name for p in family.glob("*.png")} == {f"{c.replace(' ', '_')}_p1.png" for c in test["canonical_wound_label"].unique()}
