#!/usr/bin/env python3
"""Existing union-ROI and per-box models predicting the test crops of another manifest (T19 primary analysis).

For each of the 8 families this loads the model behind the scenario's reported predictions and
predicts the test rows of ``--manifest`` (auto-box crops):
- single networks: the selected learning-rate run's ``best.pt``, as ``gradcam.checkpoint_folder`` finds it;
- AlexNet ensemble: all three members, soft-voted with equal weights;
- CNN-SVM: the chosen SVM, refit on the cached train + validation features exactly as
  ``cnn_svm.py`` fitted the reported model (that file never saved the SVM); the test crops'
  features are extracted with the same extractors.

Prediction runs as in training (bf16 autocast on GPU), so ``--check-manual`` can re-predict the
model's own manual test crops and compare them with its reported ``test_predictions.csv``.
Outputs go to ``<output-root>/<roi|box>/<family>/test_predictions.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import FAMILY_ORDER  # noqa: E402
from src.evaluation.gradcam import checkpoint_folder  # noqa: E402
from src.training.cnn_svm import extract, fit_svm  # noqa: E402
from src.training.common import prediction_frame, resolve_device  # noqa: E402
from src.training.data import ManifestImageDataset, load_manifest, prepare_rows  # noqa: E402
from src.training.engine import load_checkpoint, predict  # noqa: E402
from src.training.models import build_model, feature_extractor  # noqa: E402
from src.training.photo import PHOTO_INPUT_PATHS, manifest_photos, photo_weights  # noqa: E402
from src.training.transforms import build_transforms  # noqa: E402


DESIGN_PATHS = {"union": "roi", "box": "box"}


def manifest_rows(config: Dict[str, Any], input_path: str) -> pd.DataFrame:
    """The rows a run was trained and tested on: its own manifest with its scope and smoke options."""
    return load_manifest(Path(config["manifest"]), input_path, config.get("full_scope", "matched"),
                         Path(config.get("roi_manifest", "data/roi/roi_manifest.csv")), bool(config.get("smoke", False)))


def test_rows(manifest: Path, input_path: str, image_root: Path, classes: List[str],
              df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # Resolve image paths and labels; the manifest's classes must match the model's, then keep test
    df = load_manifest(manifest, input_path) if df is None else df
    df, found = prepare_rows(df, input_path, image_root)
    if found != classes:
        raise ValueError(f"{manifest}: classes {found} differ from the model's {classes}")
    return df[df["split"] == "test"]


def loader_for(rows: pd.DataFrame, image_size: int, cache: bool, batch_size: int, workers: int) -> DataLoader:
    # Same evaluation transforms and image cache as when the model was trained
    dataset = ManifestImageDataset(rows["image_path"].tolist(), rows["label_index"].tolist(),
                                   build_transforms(image_size, False),
                                   cache_side=2 * image_size if cache else None, threads=max(workers, 4))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def network_probs(folder: Path, family: str, rows: pd.DataFrame, device: torch.device, workers: int) -> np.ndarray:
    # Rebuild the network, load its best checkpoint and predict (bf16 on GPU, as in the reported run)
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    # Build it as the run did: torchvision's GoogLeNet turns on transform_input only with pretrained weights
    model = build_model(family, len(config["classes"]), pretrained=bool(config.get("pretrained", True)))
    model.load_state_dict(load_checkpoint(folder / "best.pt", device)["model"])
    loader = loader_for(rows, int(config["image_size"]), config.get("cache", True), int(config["batch_size"]), workers)
    _, _, probs = predict(model.to(device), loader, nn.CrossEntropyLoss(), device)
    return probs


def svm_model(folder: Path, image_root_of_run: Path, workers: int):
    """The reported CNN-SVM refit from its cached train + validation features; returns (model, config)."""
    # The rows the CNN-SVM was trained on, read exactly as cnn_svm.py read them
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    input_path = config["input_path"]
    df, classes = prepare_rows(manifest_rows(config, input_path), input_path, image_root_of_run)
    per_box = input_path in PHOTO_INPUT_PATHS
    xs, ys, ws = [], [], []
    # Load the cached features of train and validation; they must be for exactly these rows
    for split in ("train", "validation"):
        rows = df[df["split"] == split]
        parts = []
        for fam in config["extractors"]:
            with np.load(folder / "features" / f"{fam}_{split}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["relative_path"], rows["image_relative_path"].to_numpy().astype(str)):
                    raise ValueError(f"{folder}: cached {fam} {split} features do not match the manifest rows")
                parts.append(saved["features"])
        xs.append(np.concatenate(parts, axis=1))
        ys.append(rows["label_index"].to_numpy())
        ws.append(photo_weights(manifest_photos(rows)) if per_box else None)
    # Refit the chosen SVM on train + validation, exactly as the reported model was fitted
    model = fit_svm(config["best_params"], int(config["seed"]), np.vstack(xs), np.concatenate(ys),
                    np.concatenate(ws) if per_box else None, probability=True)
    return model, config, classes


def svm_probs(model, config: Dict[str, Any], rows: pd.DataFrame, device: torch.device, workers: int) -> np.ndarray:
    # Extract the new crops' features with the same CNNs, then SVM probabilities
    loader = loader_for(rows, int(config["image_size"]), config.get("cache", True), int(config["batch_size"]), workers)
    parts = []
    for fam in config["extractors"]:
        extractor, _ = feature_extractor(fam, pretrained=config.get("pretrained", True))
        parts.append(extract(extractor.to(device), loader, device))
        del extractor
    return model.predict_proba(np.concatenate(parts, axis=1))


def family_predictions(root: Path, design: str, family: str, rows: pd.DataFrame, device: torch.device,
                       workers: int) -> np.ndarray:
    input_path = DESIGN_PATHS[design]
    scenario = root / input_path / family
    # Ensemble: predict with each of the three members and average (soft vote)
    if family == "alexnet_ensemble":
        members = pd.read_csv(scenario / "ensemble_members.csv")
        return np.mean([network_probs(Path(d), "alexnet", rows, device, workers) for d in members["output_dir"]], axis=0)
    # CNN-SVM: refit, then predict
    if family == "cnn_svm":
        image_root = Path(json.loads((scenario / "run_config.json").read_text(encoding="utf-8"))["image_root"])
        model, config, _ = svm_model(scenario, image_root, workers)
        return svm_probs(model, config, rows, device, workers)
    # Single network: the learning-rate run that produced the reported predictions
    return network_probs(checkpoint_folder(root, input_path, family), family, rows, device, workers)


def config_folder(root: Path, design: str, family: str) -> Path:
    """The folder whose run_config.json describes the scenario's model (first member for the ensemble)."""
    input_path = DESIGN_PATHS[design]
    scenario = root / input_path / family
    if family == "alexnet_ensemble":
        return Path(pd.read_csv(scenario / "ensemble_members.csv")["output_dir"].iloc[0])
    if family == "cnn_svm":
        return scenario
    return checkpoint_folder(root, input_path, family)


def run_config(root: Path, design: str, family: str) -> Dict[str, Any]:
    return json.loads((config_folder(root, design, family) / "run_config.json").read_text(encoding="utf-8"))


def check_manual(root: Path, design: str, family: str, device: torch.device, workers: int) -> Dict[str, Any]:
    """Re-predict the model's own manual test crops and compare with its reported prediction file."""
    # The model's own manual test crops, read with its own manifest options
    config = run_config(root, design, family)
    classes = list(config["classes"])
    input_path = DESIGN_PATHS[design]
    rows = test_rows(Path(config["manifest"]), input_path, Path(config["image_root"]), classes,
                     manifest_rows(config, input_path))
    # Predict them again and compare with the reported file, crop by crop
    probs = family_predictions(root, design, family, rows, device, workers)
    reported = pd.read_csv(root / DESIGN_PATHS[design] / family / "test_predictions.csv", dtype=str)
    reported = reported.set_index("relative_path").loc[rows["image_relative_path"]]
    reported_probs = reported[[f"prob_{c.replace(' ', '_')}" for c in classes]].to_numpy(float)
    return {"family": family, "design": design, "rows": int(len(rows)),
            "same_predicted_class": float((probs.argmax(1) == reported_probs.argmax(1)).mean()),
            "max_abs_probability_difference": float(np.abs(probs - reported_probs).max())}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Existing models predicting another manifest's test crops (T19).")
    parser.add_argument("--experiments-root", type=Path, required=True,
                        help="The main run (union design) or the T17 run (box design).")
    parser.add_argument("--design", choices=tuple(DESIGN_PATHS), required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="data/roi_auto/<arm>/<design>/roi_manifest.csv")
    parser.add_argument("--image-root", type=Path, required=True, help="The auto crops: derived_data/roi_auto_<arm>_<design>")
    parser.add_argument("--output-root", type=Path, required=True, help="experiments/auto_<arm>_primary")
    parser.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=list(FAMILY_ORDER))
    parser.add_argument("--check-manual", action="store_true", help="Also re-predict the manual test crops and compare.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    input_path = DESIGN_PATHS[args.design]
    checks = []
    # For each family: its classes, the auto manifest's test crops, the predictions
    for family in args.families:
        classes = list(run_config(args.experiments_root, args.design, family)["classes"])
        rows = test_rows(args.manifest, input_path, args.image_root, classes)
        probs = family_predictions(args.experiments_root, args.design, family, rows, device, args.num_workers)
        # Write the predictions and where they came from
        out = args.output_root / input_path / family
        out.mkdir(parents=True, exist_ok=True)
        frame = prediction_frame(rows, rows["label_index"].to_numpy(), probs, classes, photo=args.design == "box")
        frame.to_csv(out / "test_predictions.csv", index=False)
        (out / "source.json").write_text(json.dumps({"experiments_root": str(args.experiments_root),
                                                     "manifest": str(args.manifest), "image_root": str(args.image_root),
                                                     "design": args.design, "family": family}, indent=2) + "\n")
        print(f"{family} ({args.design}): {len(frame)} test crops -> {out}", flush=True)
        # Optional: prove the reloaded model reproduces its reported predictions
        if args.check_manual:
            checks.append(check_manual(args.experiments_root, args.design, family, device, args.num_workers))
            print(f"  manual re-prediction: {checks[-1]}", flush=True)
    if checks:
        (args.output_root / f"check_manual_{args.design}.json").write_text(json.dumps(checks, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
