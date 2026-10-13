#!/usr/bin/env python3
"""Grad-CAM for every fine-tuned family on one fixed set of test images, full and ROI (T14).

The same 2 test images per class are explained for every family and both input paths, with the
checkpoint that produced each scenario's reported predictions. For full images the share of the
map inside the annotated wound box is reported next to the box's share of the image. Outputs are
overlays on patient photographs and stay with the run on Raptor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import transforms as T  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_roi_dataset import find_source_image  # noqa: E402
from src.training.common import resolve_device  # noqa: E402
from src.training.data import GROUP_COLUMN, LABEL_COLUMN, load_image  # noqa: E402
from src.training.engine import load_checkpoint  # noqa: E402
from src.training.models import build_model  # noqa: E402
from src.training.transforms import IMAGENET_MEAN, IMAGENET_STD, PadToSquare  # noqa: E402


FAMILIES = ("efficientnet_b0", "resnet152", "mobilenet_v3_small", "googlenet", "vgg16", "vit_b16",
            "alexnet_ensemble")  # CNN-SVM excluded (T14 §2)
TARGET_LAYERS = {
    "efficientnet_b0": lambda m: m.features[-1],
    "mobilenet_v3_small": lambda m: m.features[-1],
    "resnet152": lambda m: m.layer4[-1],
    "googlenet": lambda m: m.inception5b,
    "vgg16": lambda m: m.features[29],
    "alexnet": lambda m: m.features[11],
    "vit_b16": lambda m: m.encoder.layers[-1].ln_1,
}
INPUT_PATHS = ("full", "roi")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM on a fixed set of test images (T14).")
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, default=Path("data/roi/roi_manifest.csv"))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Build backbones without ImageNet weights before loading the checkpoint (tests).")
    return parser.parse_args(argv)


def draw_examples(roi_manifest: Path, per_class: int, seed: int) -> pd.DataFrame:
    """``per_class`` test images per class from distinct cases, drawn once with ``seed``."""
    rows = pd.read_csv(roi_manifest, dtype=str, keep_default_na=False)
    rows = rows[rows["split"].str.strip().str.lower() == "test"]
    rng = np.random.default_rng(seed)
    picked = []
    for label in sorted(rows[LABEL_COLUMN].unique()):
        of_class = rows[rows[LABEL_COLUMN] == label].sort_values("roi_relative_path")
        cases = sorted(of_class[GROUP_COLUMN].unique(), key=lambda c: (len(c), c))
        for k, case in enumerate(rng.choice(cases, size=min(per_class, len(cases)), replace=False), start=1):
            picked.append({**of_class[of_class[GROUP_COLUMN] == case].iloc[0].to_dict(), "set": k})
    return pd.DataFrame(picked).reset_index(drop=True)


def checkpoint_folder(experiments: Path, input_path: str, family: str) -> Path:
    """Folder of the run whose best.pt produced the scenario's reported predictions."""
    scenario = experiments / input_path / family
    if family == "alexnet_ensemble":
        members = pd.read_csv(scenario / "ensemble_members.csv")
        return Path(members.loc[members["representative"].astype(str) == "True", "output_dir"].item())
    if (scenario / "tuning_summary.csv").is_file():
        summary = pd.read_csv(scenario / "tuning_summary.csv")
        return Path(summary.loc[summary["selected"].astype(str) == "True", "output_dir"].item())
    return scenario


def tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    """ViT tokens (B, 1 + N, C) -> (B, C, sqrt(N), sqrt(N)), class token dropped."""
    patches = tokens[:, 1:, :]
    side = int(math.isqrt(patches.shape[1]))
    return patches.reshape(patches.shape[0], side, side, patches.shape[2]).permute(0, 3, 1, 2)


def grad_cam(model: nn.Module, family: str, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Grad-CAM for the predicted class, upsampled to ``x`` and scaled to [0, 1]; and the probabilities."""
    store: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = TARGET_LAYERS["alexnet" if family == "alexnet_ensemble" else family](model).register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        logits[0, int(probs.argmax())].backward()
    finally:
        handle.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    if family == "vit_b16":
        act, grad = tokens_to_map(act), tokens_to_map(grad)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    peak = cam.max()
    cam = cam / peak if peak > 0 else torch.zeros_like(cam)
    return cam.cpu().numpy(), probs[0].detach().cpu().numpy()


def load_model(folder: Path, family: str, num_classes: int, pretrained: bool, device: torch.device) -> nn.Module:
    model = build_model("alexnet" if family == "alexnet_ensemble" else family, num_classes, pretrained)
    model.load_state_dict(load_checkpoint(folder / "best.pt", device)["model"])
    for module in model.modules():  # in-place activations would overwrite the hooked tensors
        if hasattr(module, "inplace"):
            module.inplace = False
    return model.to(device).eval()


def wound_box(row: Dict[str, Any], size: int) -> Optional[Tuple[float, float, float, float]]:
    """The manifest's padded union box in the padded, resized model input (full path)."""
    try:
        x1, y1, x2, y2, w, h = (float(row[k]) for k in ("x1", "y1", "x2", "y2", "source_width", "source_height"))
    except (KeyError, ValueError):
        return None
    side = max(w, h)
    dx, dy, s = (side - w) / 2, (side - h) / 2, size / side
    return ((x1 + dx) * s, (y1 + dy) * s, (x2 + dx) * s, (y2 + dy) * s)


def mass_in_box(cam: np.ndarray, box) -> Tuple[float, float]:
    if box is None or cam.sum() <= 0:
        return float("nan"), float("nan")
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return float(cam[y1:y2, x1:x2].sum() / cam.sum()), float((x2 - x1) * (y2 - y1) / cam.size)


def explain(experiments: Path, family: str, input_path: str, examples: pd.DataFrame,
            pretrained: bool, device: torch.device) -> List[Dict[str, Any]]:
    folder = checkpoint_folder(experiments, input_path, family)
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    classes, size, root = config["classes"], int(config["image_size"]), Path(config["image_root"])
    model = load_model(folder, family, len(classes), pretrained, device)
    show = T.Compose([PadToSquare(), T.Resize((size, size))])
    to_input = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    reported = pd.read_csv(folder / "test_predictions.csv", dtype=str).set_index("relative_path")["predicted_class"]

    results = []
    for row in examples.to_dict("records"):
        path = root / row["roi_relative_path"] if input_path == "roi" else find_source_image(root, row)
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"{input_path} image for {row['roi_relative_path']} not found under {root}")
        # Same pixels as prediction: the in-memory cache shrinks the longest side to 2 x size first.
        image = show(load_image(Path(path), 2 * size if config.get("cache", True) else None))
        cam, probs = grad_cam(model, family, to_input(image).unsqueeze(0).to(device))
        predicted = classes[int(probs.argmax())]
        key = Path(path).relative_to(root).as_posix()
        in_box, box_share = mass_in_box(cam, wound_box(row, size)) if input_path == "full" else (np.nan, np.nan)
        results.append({
            "family": family, "input_path": input_path, "set": row["set"], "class": row[LABEL_COLUMN],
            "case": row[GROUP_COLUMN], "image": key, "predicted_class": predicted,
            "probability": float(probs.max()), "correct": predicted == row[LABEL_COLUMN],
            "matches_prediction_file": reported.get(key) == predicted,
            "cam_in_box": in_box, "box_share": box_share, "checkpoint": str(folder / "best.pt"),
            "_image": np.asarray(image), "_cam": cam,
            "_box": wound_box(row, size) if input_path == "full" else None,
        })
    return results


def mean_of(results: List[Dict[str, Any]], key: str) -> float:
    values = np.array([r[key] for r in results], dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def plot_family(results: List[Dict[str, Any]], family: str, set_index: int, path: Path) -> None:
    """One row per class: full input, full Grad-CAM, ROI input, ROI Grad-CAM."""
    rows = sorted({r["class"] for r in results})
    by_key = {(r["class"], r["input_path"]): r for r in results if r["set"] == set_index}
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.4, 2.15 * len(rows)), squeeze=False)
    for i, label in enumerate(rows):
        for j, input_path in enumerate(INPUT_PATHS):
            r = by_key.get((label, input_path))
            image_ax, cam_ax = axes[i, 2 * j], axes[i, 2 * j + 1]
            for ax in (image_ax, cam_ax):
                ax.set_xticks([]), ax.set_yticks([])
            if r is None:
                continue
            image_ax.imshow(r["_image"])
            cam_ax.imshow(r["_image"])
            cam_ax.imshow(r["_cam"], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            if r["_box"] is not None:
                x1, y1, x2, y2 = r["_box"]
                cam_ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white",
                                           linewidth=1, linestyle="--"))
            mark = "✔" if r["correct"] else "✘"
            cam_ax.set_title(f"{mark} {r['predicted_class']} {r['probability']:.2f}", fontsize=7)
            image_ax.set_title(f"{input_path}: {label}" if j == 0 else input_path, fontsize=7)
    fig.suptitle(f"{family} Grad-CAM, test set {set_index} (box: annotated wound)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.experiments_root / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    examples = draw_examples(args.roi_manifest, args.per_class, args.seed)
    examples.to_csv(out / "examples.csv", index=False)

    table = []
    for family in args.families:
        results = [r for p in INPUT_PATHS
                   for r in explain(args.experiments_root, family, p, examples, args.pretrained, device)]
        for k in sorted(examples["set"].unique()):
            plot_family(results, family, int(k), out / f"{family}_set{k}.png")
        table += [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        full = [r for r in results if r["input_path"] == "full"]
        print(f"{family}: {sum(r['correct'] for r in results)}/{len(results)} correct, "
              f"{sum(r['matches_prediction_file'] for r in results)}/{len(results)} match prediction files, "
              f"full-image CAM in wound box {mean_of(full, 'cam_in_box'):.2f} "
              f"(box area {mean_of(full, 'box_share'):.2f})", flush=True)
    pd.DataFrame(table).to_csv(out / "gradcam_results.csv", index=False)
    print(f"Wrote examples.csv, gradcam_results.csv and figures to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
