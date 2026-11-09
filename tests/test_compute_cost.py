"""Tests for the computational cost table (T33): small images, CPU, no downloads."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from torch import nn

from src.evaluation.compute_cost import build, count_flops, count_parameters, main as cost_main, svm_cost
from src.training.cnn_svm import main as svm_main
from tests.synthetic_data import build_synthetic_dataset


@pytest.fixture(scope="module")
def svm_run(tmp_path_factory):
    # A finished CNN-SVM smoke run on synthetic ROI crops, with its cached features
    data = build_synthetic_dataset(tmp_path_factory.mktemp("wounds"))
    out = tmp_path_factory.mktemp("cnn_svm")
    svm_main(["--input-path", "roi", "--manifest", str(data["roi_manifest"]), "--image-root", str(data["roi_root"]),
              "--roi-manifest", str(data["roi_manifest"]), "--output-dir", str(out), "--image-size", "64",
              "--no-pretrained", "--device", "cpu", "--num-workers", "0", "--smoke"])
    return out


def test_flops_count_a_multiply_add_as_two():
    # 1x1 convolution, 3 -> 4 channels on a 2 x 2 image: 3 * 4 * 4 multiply-adds
    assert count_flops(nn.Conv2d(3, 4, 1, bias=False), 2) == 2 * 3 * 4 * 4


def test_ensemble_costs_three_members():
    single, ensemble = build("alexnet_ensemble").members[0], build("alexnet_ensemble")
    assert count_parameters(ensemble) == 3 * count_parameters(single)
    assert count_flops(ensemble, 64) == 3 * count_flops(single, 64)


def test_svm_refit_reproduces_the_run(svm_run):
    cost = svm_cost(svm_run, batch=4, repeats=2)
    # Every support vector is a training or validation row, with all three extractors' features
    assert 0 < cost["support_vectors"] <= 14 * 2
    assert cost["feature_dim"] == 4096 + 2048 + 1280
    assert cost["svm_flops"] == cost["support_vectors"] * cost["feature_dim"] * (3 if "rbf" in cost["svm"] else 2)


def test_cost_table(svm_run, tmp_path):
    cost_main(["--families", "mobilenet_v3_small", "alexnet_ensemble", "cnn_svm", "--svm-runs", str(svm_run),
               "--output-dir", str(tmp_path), "--device", "cpu", "--image-size", "64",
               "--repeats", "2", "--cpu-repeats", "2", "--cpu-threads", "2"])
    table = pd.read_csv(tmp_path / "compute_cost.csv").set_index("family")
    assert list(table.index) == ["mobilenet_v3_small", "alexnet_ensemble", "cnn_svm"]
    assert (table["input_size"] == 64).all()
    assert (table[["parameters", "flops", "cpu_latency_ms"]] > 0).all().all()
    # CNN-SVM: the SVM's FLOPs are added to its three extractors'
    svm = pd.read_csv(tmp_path / "svm_cost.csv").iloc[0]
    assert table.loc["cnn_svm", "flops"] == count_flops(build("cnn_svm"), 64) + svm["svm_flops"]
    assert (tmp_path / "compute_cost.json").is_file()
