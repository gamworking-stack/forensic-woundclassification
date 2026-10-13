#!/usr/bin/env python3
"""Hybrid CNN-SVM: frozen ImageNet features from several CNNs, concatenated, into an SVM.

The SVM is tuned on validation macro-F1 with the scaler and SVM fit on train only. The chosen
setting is refit on train+validation and the test split is predicted once. Test features are
extracted only after the SVM has been chosen.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.common import (  # noqa: E402
    Logger,
    environment_info,
    git_state,
    prediction_frame,
    resolve_device,
)
from src.training.config import FAMILIES  # noqa: E402
from src.training.data import (  # noqa: E402
    GROUP_COLUMN,
    SPLIT_NAMES,
    ManifestImageDataset,
    load_manifest,
    prepare_rows,
)
from src.training.engine import classification_metrics, seed_everything  # noqa: E402
from src.training.models import feature_extractor, input_size  # noqa: E402
from src.training.transforms import build_transforms  # noqa: E402


DEFAULT_EXTRACTORS = ("vgg16", "resnet152", "efficientnet_b0")
EXTRACTOR_CHOICES = tuple(f for f in FAMILIES if f != "vit_b16")  # one shared 384 px input
RBF_C = (0.1, 1.0, 10.0, 100.0, 1000.0)
RBF_GAMMA = ("scale", 1e-6, 1e-5, 1e-4, 1e-3)
LINEAR_C = (0.001, 0.01, 0.1, 1.0)


def svm_grid() -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = [{"kernel": "rbf", "C": c, "gamma": g} for c in RBF_C for g in RBF_GAMMA]
    grid += [{"kernel": "linear", "C": c} for c in LINEAR_C]
    return grid


def grid_edges(params: Dict[str, Any]) -> List[str]:
    """Grid limits the chosen setting sits on; a best value on the edge means the grid may be too narrow."""
    c_values = RBF_C if params["kernel"] == "rbf" else LINEAR_C
    edges = [f"C={params['C']}" for limit in (min(c_values), max(c_values)) if params["C"] == limit]
    numeric_gammas = [g for g in RBF_GAMMA if g != "scale"]
    if params["kernel"] == "rbf" and params["gamma"] in (min(numeric_gammas), max(numeric_gammas)):
        edges.append(f"gamma={params['gamma']}")
    return edges


def make_svm(params: Dict[str, Any], seed: int, probability: bool = False) -> Pipeline:
    """Scaler and SVM in one pipeline, so the scaler only ever sees the SVM's training data."""
    svm = SVC(**params, class_weight="balanced", probability=probability, random_state=seed)
    return Pipeline([("scale", StandardScaler()), ("svm", svm)])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid CNN-SVM on one input path (full image or ROI).")
    parser.add_argument("--input-path", required=True, choices=("full", "roi"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, default=Path("data/roi/roi_manifest.csv"))
    parser.add_argument("--full-scope", choices=("matched", "all"), default="matched")
    parser.add_argument("--extractors", nargs="+", choices=EXTRACTOR_CHOICES, default=list(DEFAULT_EXTRACTORS))
    parser.add_argument("--image-size", type=int, default=None, help="Override the 384 px input (tests only).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--smoke", action="store_true", help="2 images per class per split.")
    parser.add_argument("--cv-folds", type=Path, default=None,
                        help="CV run (T12b): fold table; test rows are dropped and --fold is validation.")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--svm-params-from", type=Path, default=None,
                        help="run_config.json of a finished CNN-SVM run; fit its best_params instead of the grid.")
    args = parser.parse_args(argv)
    if (args.cv_folds is None) != (args.fold is None):
        parser.error("--cv-folds and --fold go together")
    return args


@torch.no_grad()
def extract(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    features = []
    for images, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images.to(device, non_blocking=True))
        features.append(output.float().cpu().numpy())
    return np.concatenate(features)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    feature_dir = out / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "train.log")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    image_size = args.image_size or input_size(args.extractors[0])

    df = load_manifest(args.manifest, args.input_path, args.full_scope, args.roi_manifest, args.smoke,
                       args.cv_folds, args.fold)
    cv = args.cv_folds is not None
    df, classes = prepare_rows(df, args.input_path, args.image_root)
    splits = ("train", "validation") if cv else SPLIT_NAMES
    rows = {split: df[df["split"] == split] for split in splits}
    labels = {split: rows[split]["label_index"].to_numpy() for split in splits}

    run_config: Dict[str, Any] = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "family": "cnn_svm",
        "image_size": image_size,
        "classes": classes,
        "svm_grid": [] if args.svm_params_from else svm_grid(),
        "images_per_split": {s: len(rows[s]) for s in splits},
        "cases_per_split": {s: int(rows[s][GROUP_COLUMN].nunique()) for s in splits},
        **git_state(),
        "environment": environment_info(device),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    def write_config() -> None:
        with (out / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False)

    write_config()
    log(f"cnn_svm ({'+'.join(args.extractors)}) | {args.input_path} | {device} | {image_size}px | "
        f"images {run_config['images_per_split']} | cases {run_config['cases_per_split']}")

    def split_features(split: str) -> np.ndarray:
        """Concatenated features for one split, reusing cached files that match its rows."""
        split_rows = rows[split]
        expected_paths = split_rows["image_relative_path"].to_numpy()
        cache_files = {fam: feature_dir / f"{fam}_{split}.npz" for fam in args.extractors}

        def cached(fam: str) -> bool:
            if not cache_files[fam].exists():
                return False
            with np.load(cache_files[fam], allow_pickle=False) as saved:
                return np.array_equal(saved["relative_path"], expected_paths)

        todo = [fam for fam in args.extractors if not cached(fam)]
        if todo:
            dataset = ManifestImageDataset(
                split_rows["image_path"].tolist(),
                labels[split].tolist(),
                build_transforms(image_size, train=False),
                cache_side=2 * image_size if args.cache else None,
                threads=max(args.num_workers, 4),
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")
            for fam in todo:
                log(f"Extracting {fam} features for {split}")
                model, _ = feature_extractor(fam, args.pretrained)
                features = extract(model.to(device), loader, device)
                np.savez(cache_files[fam], features=features, relative_path=expected_paths.astype(str))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        parts = []
        for fam in args.extractors:
            with np.load(cache_files[fam], allow_pickle=False) as saved:
                parts.append(saved["features"])
        return np.concatenate(parts, axis=1)

    x_train, x_val = split_features("train"), split_features("validation")
    run_config["feature_dim"] = int(x_train.shape[1])
    log(f"Features: train {x_train.shape}, validation {x_val.shape}")

    if args.svm_params_from:
        best = json.loads(args.svm_params_from.read_text(encoding="utf-8"))["best_params"]
        run_config["best_params"] = best
        log(f"Fixed SVM {best} from {args.svm_params_from} (no grid)")
    else:
        best = tune(x_train, x_val, labels, classes, args.seed, out, log, run_config)

    val_model = make_svm(best, args.seed, probability=True).fit(x_train, labels["train"])
    prediction_frame(rows["validation"], labels["validation"], val_model.predict_proba(x_val), classes) \
        .to_csv(out / "validation_predictions.csv", index=False)
    if cv:  # held-out fold predicted; the test split is never loaded
        write_config()
        log(f"Wrote validation predictions to {out} (CV fold {args.fold})")
        return 0

    # The test split is touched for the first time here, after the SVM has been chosen.
    x_test = split_features("test")
    final = make_svm(best, args.seed, probability=True).fit(
        np.vstack([x_train, x_val]), np.concatenate([labels["train"], labels["validation"]])
    )
    prediction_frame(rows["test"], labels["test"], final.predict_proba(x_test), classes) \
        .to_csv(out / "test_predictions.csv", index=False)

    write_config()
    log(f"Wrote predictions to {out} (test from SVM refit on train+validation)")
    return 0


def tune(x_train, x_val, labels, classes, seed, out, log, run_config) -> Dict[str, Any]:
    """Validation macro-F1 for every grid setting; returns the chosen one and logs grid-edge warnings."""
    grid = svm_grid()
    tuning = []
    for params in grid:
        pred = make_svm(params, seed).fit(x_train, labels["train"]).predict(x_val)
        metrics = classification_metrics(labels["validation"], pred, len(classes))
        tuning.append({
            "kernel": params["kernel"],
            "C": params["C"],
            "gamma": params.get("gamma", ""),
            "validation_macro_f1": metrics["macro_f1"],
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
        })
        log(json.dumps(tuning[-1]))
    pd.DataFrame(tuning).to_csv(out / "svm_validation_tuning.csv", index=False)

    # Highest validation macro-F1; ties go to the earlier grid entry.
    best_index = max(range(len(grid)), key=lambda i: (tuning[i]["validation_macro_f1"], -i))
    best = grid[best_index]
    run_config["best_params"] = best
    run_config["best_validation_macro_f1"] = tuning[best_index]["validation_macro_f1"]
    log(f"Chosen SVM {best} (validation macro-F1={tuning[best_index]['validation_macro_f1']:.6f})")
    edges = grid_edges(best)
    if edges:
        log(f"WARNING: chosen SVM is on the edge of the grid ({', '.join(edges)}); consider widening it")

    return best


if __name__ == "__main__":
    sys.exit(main())
