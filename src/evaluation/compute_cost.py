#!/usr/bin/env python3
"""Computational cost of every model family (T33): size, FLOPs, inference speed, training memory.

Each family is built as it is trained (``build_model``, 7 classes) at its own input size; the
weights do not change the cost, so no checkpoint is loaded. Per family:

- parameters (all are fine-tuned);
- FLOPs of one image (PyTorch's FlopCounterMode: convolutions and matrix products, a
  multiply-add counted as 2 FLOPs);
- GPU latency of one image and throughput in batches of 32 (bfloat16 autocast, as ``predict``),
  and CPU latency of one image (float32); the model only, decoding and resizing are not timed;
- peak GPU memory of training steps as ``train_one_epoch`` takes them (batch 32, bfloat16
  autocast, AdamW), weights, gradients and optimiser state included.

The AlexNet ensemble is its 3 members run one after the other; members are trained one at a time,
so its training memory is one member's. CNN-SVM is its 3 frozen extractors plus the SVM, refitted
from a finished run's cached features exactly as ``cnn_svm.py`` fits its final model; it trains
no network, so its memory is the peak of feature extraction (one extractor at a time, batch 64).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import FAMILY_ORDER  # noqa: E402
from src.training.cnn_svm import DEFAULT_EXTRACTORS, fit_svm  # noqa: E402
from src.training.common import environment_info, resolve_device  # noqa: E402
from src.training.data import load_manifest, prepare_rows  # noqa: E402
from src.training.engine import _autocast, train_one_epoch  # noqa: E402
from src.training.models import build_model, feature_extractor, input_size  # noqa: E402
from src.training.photo import PHOTO_INPUT_PATHS, manifest_photos, photo_weights  # noqa: E402

NUM_CLASSES = 7
ENSEMBLE_MEMBERS = 3
GIB = 1024 ** 3


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameters, FLOPs, latency, throughput and training memory.")
    parser.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=list(FAMILY_ORDER))
    parser.add_argument("--svm-runs", type=Path, nargs="*", default=[],
                        help="Finished CNN-SVM run folders (run_config.json, features/) whose SVM is refitted.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, default=None, help="Override every input size (tests only).")
    parser.add_argument("--batch-size", type=int, default=32, help="Throughput and training batch (as train.py).")
    parser.add_argument("--svm-batch-size", type=int, default=64, help="CNN-SVM extraction batch (as cnn_svm.py).")
    parser.add_argument("--repeats", type=int, default=200, help="Timed GPU passes of one image.")
    parser.add_argument("--cpu-repeats", type=int, default=30, help="Timed CPU passes of one image.")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=3, help="Training steps for the memory peak.")
    return parser.parse_args(argv)


class Ensemble(nn.Module):
    """Mean of the members' class probabilities (soft vote, as ensemble.py)."""

    def __init__(self, members: List[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x).float().softmax(dim=1) for m in self.members]).mean(dim=0)


class Concatenated(nn.Module):
    """CNN-SVM's frozen extractors, their feature vectors side by side."""

    def __init__(self, extractors: List[nn.Module]) -> None:
        super().__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([e(x).float() for e in self.extractors], dim=1)


def build(family: str) -> nn.Module:
    """The network of one family as used at prediction time (random weights)."""
    if family == "alexnet_ensemble":
        return Ensemble([build_model("alexnet", NUM_CLASSES, pretrained=False) for _ in range(ENSEMBLE_MEMBERS)])
    if family == "cnn_svm":
        return Concatenated([feature_extractor(f, pretrained=False)[0] for f in DEFAULT_EXTRACTORS])
    return build_model(family, NUM_CLASSES, pretrained=False)


def family_size(family: str, override: Optional[int]) -> int:
    if override:
        return override
    # Composite models take the input size of their parts (all 384 px)
    part = {"alexnet_ensemble": "alexnet", "cnn_svm": DEFAULT_EXTRACTORS[0]}.get(family, family)
    return input_size(part)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, size: int) -> int:
    """FLOPs of one image (float32, CPU; the count does not depend on the device)."""
    model = model.cpu().eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.randn(1, 3, size, size))
    return int(counter.get_total_flops())


def timed(run: Callable[[], None], repeats: int, warmup: int, device: torch.device) -> List[float]:
    """Seconds per call, each call synchronised on GPU so it is timed to completion."""
    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        run()
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        sync()
        times.append(time.perf_counter() - start)
    return times


def quartiles(times: List[float]) -> Dict[str, float]:
    # Median and interquartile range in milliseconds
    q1, median, q3 = np.percentile(np.array(times) * 1000, [25, 50, 75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3)}


@torch.inference_mode()
def inference_speed(model: nn.Module, size: int, batch: Optional[int], repeats: int,
                    device: torch.device) -> Dict[str, float]:
    """One-image latency and, with ``batch``, batch throughput on ``device`` (bfloat16 autocast on GPU)."""
    model = model.to(device).eval()
    one = torch.randn(1, 3, size, size, device=device)

    def forward(x: torch.Tensor) -> Callable[[], None]:
        def run() -> None:
            with _autocast(device):
                model(x)
        return run

    latency = quartiles(timed(forward(one), repeats, max(repeats // 10, 3), device))
    speed = {"latency_ms": latency["median"], "latency_q1_ms": latency["q1"], "latency_q3_ms": latency["q3"]}
    if batch:
        many = torch.randn(batch, 3, size, size, device=device)
        speed["throughput_images_per_s"] = batch / statistics.median(timed(forward(many), max(repeats // 4, 5), 5, device))
    return speed


def training_memory(family: str, size: int, batch: int, steps: int, device: torch.device) -> Dict[str, float]:
    """Peak GPU memory of ``steps`` training steps of one network as train.py takes them."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model("alexnet" if family == "alexnet_ensemble" else family, NUM_CLASSES, pretrained=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.ones(NUM_CLASSES, device=device), label_smoothing=0.05)
    batches = [(torch.randn(batch, 3, size, size, device=device),
                torch.randint(0, NUM_CLASSES, (batch,), device=device)) for _ in range(steps)]
    train_one_epoch(model, batches, criterion, optimizer, device)
    torch.cuda.synchronize(device)
    peak = {"train_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
            "train_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB}
    del model, optimizer, batches
    torch.cuda.empty_cache()
    return peak


@torch.inference_mode()
def extraction_memory(size: int, batch: int, device: torch.device) -> Dict[str, float]:
    """CNN-SVM: peak GPU memory of feature extraction, one extractor at a time (as cnn_svm.py)."""
    peaks = []
    for family in DEFAULT_EXTRACTORS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = feature_extractor(family, pretrained=False)[0].to(device).eval()
        with _autocast(device):
            model(torch.randn(batch, 3, size, size, device=device))
        torch.cuda.synchronize(device)
        peaks.append((torch.cuda.max_memory_allocated(device) / GIB, torch.cuda.max_memory_reserved(device) / GIB))
        del model
    torch.cuda.empty_cache()
    return {"train_peak_allocated_gib": max(p[0] for p in peaks),
            "train_peak_reserved_gib": max(p[1] for p in peaks)}


def refit_svm(run_dir: Path, seed: int = 42):
    """The final SVM of a finished CNN-SVM run, refitted on its cached train+validation features."""
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    df = load_manifest(Path(config["manifest"]), config["input_path"], config.get("full_scope", "matched"),
                       Path(config["roi_manifest"]), config.get("smoke", False))
    df, _ = prepare_rows(df, config["input_path"], Path(config["image_root"]))
    parts, labels, weights = [], [], []
    for split in ("train", "validation"):
        rows = df[df["split"] == split]
        features = []
        for family in config["extractors"]:
            with np.load(run_dir / "features" / f"{family}_{split}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["relative_path"], rows["image_relative_path"].to_numpy()):
                    raise ValueError(f"{run_dir} {family} {split} features do not match the manifest rows")
                features.append(saved["features"])
        parts.append(np.concatenate(features, axis=1))
        labels.append(rows["label_index"].to_numpy())
        if config["input_path"] in PHOTO_INPUT_PATHS:
            weights.append(photo_weights(manifest_photos(rows)))
    x, y = np.vstack(parts), np.concatenate(labels)
    svm = fit_svm(config["best_params"], config.get("seed", seed), x, y,
                  np.concatenate(weights) if weights else None, probability=True)
    return svm, x, config


def svm_cost(run_dir: Path, batch: int, repeats: int) -> Dict[str, object]:
    """Support vectors, FLOPs per image (kernel against every support vector) and CPU time of the SVM."""
    svm, x, config = refit_svm(run_dir)
    vectors, dim = svm.named_steps["svm"].support_vectors_.shape
    # RBF: difference, square and sum per feature (3 FLOPs); linear: a multiply-add (2 FLOPs)
    per_feature = 3 if config["best_params"]["kernel"] == "rbf" else 2
    one, many = x[:1], x[:batch]
    latency = quartiles(timed(lambda: svm.predict_proba(one), repeats, 3, torch.device("cpu")))
    batch_time = statistics.median(timed(lambda: svm.predict_proba(many), max(repeats // 4, 5), 2, torch.device("cpu")))
    return {"run": str(run_dir), "input_path": config["input_path"], "svm": json.dumps(config["best_params"]),
            "support_vectors": int(vectors), "feature_dim": int(dim),
            "svm_flops": int(vectors * dim * per_feature), "svm_latency_ms": latency["median"],
            "svm_batch_ms": batch_time * 1000}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.backends.cudnn.benchmark = False  # as training and prediction (seed_everything)
    torch.set_num_threads(args.cpu_threads)

    # The SVM part of CNN-SVM, one row per finished run (full image, per crop)
    svms = [svm_cost(run, args.batch_size, args.cpu_repeats) for run in args.svm_runs]
    if svms:
        pd.DataFrame(svms).to_csv(out / "svm_cost.csv", index=False)
    svm = svms[0] if svms else None

    rows = []
    for family in args.families:
        size = family_size(family, args.image_size)
        model = build(family)
        row = {"family": family, "input_size": size, "parameters": count_parameters(model),
               "flops": count_flops(model, size)}
        # Inference: GPU in bfloat16 when there is one, and always CPU in float32
        if device.type == "cuda":
            gpu = inference_speed(model, size, args.batch_size, args.repeats, device)
            row.update({f"gpu_{k}": v for k, v in gpu.items()})
        cpu = inference_speed(model, size, None, args.cpu_repeats, torch.device("cpu"))
        row.update({f"cpu_{k}": v for k, v in cpu.items()})
        # CNN-SVM: the first SVM run's support vectors, FLOPs and CPU time are added to the networks'
        if family == "cnn_svm" and svm is not None:
            row.update({"svm_support_vectors": svm["support_vectors"], "flops": row["flops"] + svm["svm_flops"]})
            if device.type == "cuda":
                row["gpu_latency_ms"] += svm["svm_latency_ms"]
                row["gpu_throughput_images_per_s"] = args.batch_size / (
                    args.batch_size / row["gpu_throughput_images_per_s"] + svm["svm_batch_ms"] / 1000)
            row["cpu_latency_ms"] += svm["svm_latency_ms"]
        del model
        # Training memory: one network trained as train.py does; CNN-SVM only extracts features
        if device.type == "cuda":
            row.update(extraction_memory(size, args.svm_batch_size, device) if family == "cnn_svm"
                       else training_memory(family, size, args.batch_size, args.train_steps, device))
        rows.append(row)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)

    pd.DataFrame(rows).to_csv(out / "compute_cost.csv", index=False)
    settings = {**vars(args), "environment": environment_info(device), "cpu_threads": torch.get_num_threads()}
    (out / "compute_cost.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    print(f"Wrote compute_cost.csv ({len(rows)} families) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
