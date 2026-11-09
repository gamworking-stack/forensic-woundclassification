"""Tests for the generator dataset export (T34): training crops only, square, class-labelled."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from PIL import Image

from src.data.export_generator_crops import REPO_ROOT, main as export_main
from tests.synthetic_data import build_synthetic_box_dataset, build_synthetic_dataset


@pytest.fixture(scope="module")
def boxes(tmp_path_factory):
    root = tmp_path_factory.mktemp("wounds")
    return build_synthetic_box_dataset(root, build_synthetic_dataset(root))


def test_exports_training_crops_only(boxes, tmp_path):
    export_main(["--manifest", str(boxes["box_manifest"]), "--image-root", str(boxes["box_root"]),
                 "--output-dir", str(tmp_path), "--size", "32"])
    manifest = pd.read_csv(boxes["box_manifest"])
    train = manifest[manifest["split"] == "train"]
    crops = pd.read_csv(tmp_path / "crops.csv")
    # Every training crop once, nothing from validation or test
    assert sorted(crops["roi_relative_path"]) == sorted(train["roi_relative_path"])
    labels = json.loads((tmp_path / "dataset.json").read_text())["labels"]
    info = json.loads((tmp_path / "dataset_info.json").read_text())
    assert len(labels) == len(train)
    # Labels index the classifiers' sorted class list, and each file sits in its class folder
    assert info["classes"] == sorted(manifest["canonical_wound_label"].unique())
    for name, label in labels:
        assert name.startswith(f"{label}/")
        with Image.open(tmp_path / name) as image:
            assert image.size == (32, 32) and image.mode == "RGB"
    assert crops.groupby("class").size().to_dict() == info["crops_per_class"]


def test_refuses_output_inside_repo(boxes):
    with pytest.raises(ValueError, match="inside the repository"):
        export_main(["--manifest", str(boxes["box_manifest"]), "--image-root", str(boxes["box_root"]),
                     "--output-dir", str(REPO_ROOT / "gan_crops")])
