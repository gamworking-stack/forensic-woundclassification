#!/usr/bin/env python3
"""Grad-CAM for every fine-tuned family on one fixed set of test images, full and ROI (T14).

The same 2 test images per class are explained for every family and both input paths, with the
checkpoint that produced each scenario's reported predictions. For full images the share of the
map inside the annotated wound box is reported next to the box's share of the image. Outputs are
overlays on patient photographs and stay with the run on Raptor.

With ``--box-manifest`` (T17) the per-box run is explained instead: the same example photos (the
main run's ``examples.csv``), a map for every crop of each photo, and the photo prediction as the
mean of its crops' probabilities.

With ``--all-test`` (and ``--box-manifest``, ``--main-root``) every test photo of ``--classes`` is
explained instead of the examples: for every crop of a photo, the full image with that crop's box
(main run) next to the crop (per-box run), one figure per family and class in ``<exp>/gradcam_all/``.

With ``--auto-arm`` (T19 §2.9) the same models explain the auto-box crops of the example photos:
union (``--auto-manifest``, main run) with the manual crop and the auto crop side by side, or per-box
(``--box-manifest`` of the arm, T17 run). Figures go to ``<exp>/gradcam_auto_<arm>/``; for auto crops
``matches_prediction_file`` is not meaningful (the prediction files hold the manual crops).
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
from src.training.photo import MANIFEST_PHOTO_COLUMN, photo_prediction_frame  # noqa: E402
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
    parser.add_argument("--box-manifest", type=Path, default=None,
                        help="T17: explain the per-box run in --experiments-root using this crop manifest.")
    parser.add_argument("--examples", type=Path, default=None,
                        help="T17: the main run's gradcam/examples.csv, so the same photos are explained.")
    parser.add_argument("--all-test", action="store_true",
                        help="Every test photo of --classes, full image (--main-root) and each per-box crop "
                             "(--experiments-root), to <exp>/gradcam_all.")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="With --all-test: the classes to explain, e.g. 'cut wound' scratch (default: all).")
    parser.add_argument("--main-root", type=Path, default=None,
                        help="With --all-test: the main run, whose full-image checkpoints are explained.")
    parser.add_argument("--auto-arm", choices=("4Z", "4P", "4F"), default=None,
                        help="T19: explain this arm's auto-box crops of the --examples photos.")
    parser.add_argument("--auto-root", type=Path, default=None, help="T19: the arm's crop folder.")
    parser.add_argument("--auto-manifest", type=Path, default=None, help="T19: the arm's union roi_manifest.csv.")
    args = parser.parse_args(argv)
    if args.auto_arm is not None:
        if args.auto_root is None or args.examples is None or (args.auto_manifest is None) == (args.box_manifest is None):
            parser.error("--auto-arm needs --auto-root, --examples, and --auto-manifest (union) or --box-manifest (box)")
        return args
    if args.auto_root is not None or args.auto_manifest is not None:
        parser.error("--auto-root and --auto-manifest go with --auto-arm")
    if args.all_test:
        if args.box_manifest is None or args.main_root is None or args.examples is not None:
            parser.error("--all-test needs --box-manifest and --main-root, and takes no --examples")
    elif (args.box_manifest is None) != (args.examples is None):
        parser.error("--box-manifest and --examples go together")
    if not args.all_test and (args.classes is not None or args.main_root is not None):
        parser.error("--classes and --main-root go with --all-test")
    return args


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


def all_test_images(roi_manifest: Path, classes: Optional[List[str]]) -> pd.DataFrame:
    """Every test image of ``classes`` (all classes if None); ``set`` numbers the images within a class."""
    rows = pd.read_csv(roi_manifest, dtype=str, keep_default_na=False)
    rows = rows[rows["split"].str.strip().str.lower() == "test"]
    if classes is not None:
        unknown = sorted(set(classes) - set(rows[LABEL_COLUMN]))
        if unknown:
            raise ValueError(f"No test images of class {unknown}; test classes are {sorted(rows[LABEL_COLUMN].unique())}")
        rows = rows[rows[LABEL_COLUMN].isin(classes)]
    rows = rows.sort_values([LABEL_COLUMN, "roi_relative_path"]).reset_index(drop=True)
    rows["set"] = rows.groupby(LABEL_COLUMN).cumcount() + 1
    return rows


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
            pretrained: bool, device: torch.device, image_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    folder = checkpoint_folder(experiments, input_path, family)
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    classes, size, root = config["classes"], int(config["image_size"]), image_root or Path(config["image_root"])
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
    labels = sorted({r["class"] for r in results})
    by_key = {(r["class"], r["input_path"]): r for r in results if r["set"] == set_index}
    rows = [(label, {p: by_key.get((label, p)) for p in INPUT_PATHS}) for label in labels]
    plot_rows(rows, f"{family} Grad-CAM, test set {set_index} (box: annotated wound)", path)


def plot_class_crops(full: List[Dict[str, Any]], crops: List[Dict[str, Any]], family: str, label: str,
                     path: Path) -> None:
    """One row per test crop of ``label`` (--all-test): full input and Grad-CAM with the crop's box, crop and Grad-CAM."""
    full_by_photo = {r["photo"]: r for r in full if r["class"] == label}
    of_class = sorted((r for r in crops if r["class"] == label), key=lambda r: (int(r["set"]), r["image"]))
    rows = []
    for r in of_class:
        photo = "✔" if r["photo_correct"] else "✘"
        name = f"{label} #{r['set']} ({r['case']}) crop {r['crop_index']}/{r['n_crops']}, photo {photo}"
        rows.append((name, {"full": {**full_by_photo[r["photo"]], "_box": r["_full_box"]}, "box crop": r}))
    photos = len({r["photo"] for r in of_class})
    plot_rows(rows, f"{family} Grad-CAM, all {len(rows)} crops of the {photos} test photos of {label} "
                    f"(box: this crop)", path)


def plot_rows(rows: List[Tuple[str, Dict[str, Optional[Dict[str, Any]]]]], title: str, path: Path) -> None:
    """One row per (name, {input path: result}), two columns per path: input and Grad-CAM."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.4, 2.15 * len(rows)), squeeze=False)
    for i, (label, by_path) in enumerate(rows):
        for j, (input_path, r) in enumerate(by_path.items()):
            image_ax, cam_ax = axes[i, 2 * j], axes[i, 2 * j + 1]
            for ax in (image_ax, cam_ax):
                ax.set_xticks([]), ax.set_yticks([])
            if r is None:
                continue
            image_ax.imshow(r["_image"])
            cam_ax.imshow(r["_image"])
            cam_ax.imshow(r["_cam"], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            if r.get("_box") is not None:
                x1, y1, x2, y2 = r["_box"]
                cam_ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white",
                                           linewidth=1, linestyle="--"))
            mark = "✔" if r["correct"] else "✘"
            cam_ax.set_title(f"{mark} {r['predicted_class']} {r['probability']:.2f}", fontsize=7)
            image_ax.set_title(f"{input_path}: {label}" if j == 0 else input_path, fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


BOX_COLUMNS = 3  # crops shown per photo in the per-box figures


def explain_box(experiments: Path, family: str, examples: pd.DataFrame, crops: pd.DataFrame,
                pretrained: bool, device: torch.device, image_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Maps for every crop of each example photo, with crop- and photo-level predictions (T17)."""
    folder = checkpoint_folder(experiments, "box", family)
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    classes, size, root = config["classes"], int(config["image_size"]), image_root or Path(config["image_root"])
    model = load_model(folder, family, len(classes), pretrained, device)
    show = T.Compose([PadToSquare(), T.Resize((size, size))])
    to_input = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    crop_file = pd.read_csv(folder / "test_predictions.csv", dtype=str)
    reported = crop_file.set_index("relative_path")["predicted_class"]
    probs_columns = [c for c in crop_file.columns if c.startswith("prob_")]
    photo_reported = photo_prediction_frame(crop_file.astype({c: float for c in probs_columns})) \
        .set_index("relative_path")["predicted_class"]

    results = []
    for row in examples.to_dict("records"):
        photo = row[MANIFEST_PHOTO_COLUMN]
        of_photo = crops[crops[MANIFEST_PHOTO_COLUMN] == photo].sort_values("roi_relative_path")
        if of_photo.empty:
            raise ValueError(f"No per-box crops for example photo {photo!r}")
        photo_results, all_probs = [], []
        for crop in of_photo.to_dict("records"):
            path = root / crop["roi_relative_path"]
            if not path.is_file():
                raise FileNotFoundError(f"box crop {crop['roi_relative_path']} not found under {root}")
            image = show(load_image(path, 2 * size if config.get("cache", True) else None))
            cam, probs = grad_cam(model, family, to_input(image).unsqueeze(0).to(device))
            all_probs.append(probs)
            predicted = classes[int(probs.argmax())]
            key = path.relative_to(root).as_posix()
            photo_results.append({
                "family": family, "input_path": "box", "set": int(row["set"]), "class": row[LABEL_COLUMN],
                "case": row[GROUP_COLUMN], "photo": photo, "image": key, "n_crops": len(of_photo),
                "predicted_class": predicted, "probability": float(probs.max()),
                "correct": predicted == row[LABEL_COLUMN], "matches_prediction_file": reported.get(key) == predicted,
                "checkpoint": str(folder / "best.pt"), "_image": np.asarray(image), "_cam": cam,
            })
        mean = np.mean(all_probs, axis=0)
        photo_predicted = classes[int(mean.argmax())]
        for r in photo_results:
            r.update({"photo_predicted_class": photo_predicted, "photo_probability": float(mean.max()),
                      "photo_correct": photo_predicted == row[LABEL_COLUMN],
                      "matches_photo_prediction_file": photo_reported.get(photo) == photo_predicted})
        results += photo_results
    return results


def plot_box_family(results: List[Dict[str, Any]], family: str, set_index: int, path: Path) -> None:
    """One row per class: up to BOX_COLUMNS crops of the example photo, each as input and Grad-CAM."""
    rows = sorted({r["class"] for r in results})
    fig, axes = plt.subplots(len(rows), 2 * BOX_COLUMNS, figsize=(2.1 * 2 * BOX_COLUMNS, 2.15 * len(rows)), squeeze=False)
    for i, label in enumerate(rows):
        crops = [r for r in results if int(r["set"]) == set_index and r["class"] == label]
        for ax in axes[i]:
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis("off")
        for j, r in enumerate(crops[:BOX_COLUMNS]):
            image_ax, cam_ax = axes[i, 2 * j], axes[i, 2 * j + 1]
            image_ax.axis("on"), cam_ax.axis("on")
            image_ax.imshow(r["_image"])
            cam_ax.imshow(r["_image"])
            cam_ax.imshow(r["_cam"], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            mark = "✔" if r["correct"] else "✘"
            cam_ax.set_title(f"{mark} {r['predicted_class']} {r['probability']:.2f}", fontsize=7)
            if j == 0:
                photo_mark = "✔" if r["photo_correct"] else "✘"
                more = f", {r['n_crops']} crops" if r["n_crops"] > BOX_COLUMNS else ""
                image_ax.set_title(f"{label}: photo {photo_mark} {r['photo_predicted_class']}{more}", fontsize=7)
    fig.suptitle(f"{family} Grad-CAM, per-box crops, test set {set_index}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main_box(args: argparse.Namespace, device: torch.device) -> int:
    out = args.experiments_root / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    examples = pd.read_csv(args.examples, dtype=str, keep_default_na=False)
    examples.to_csv(out / "examples.csv", index=False)
    crops = pd.read_csv(args.box_manifest, dtype=str, keep_default_na=False)
    crops = crops[crops["split"].str.strip().str.lower() == "test"]
    table = []
    for family in args.families:
        results = explain_box(args.experiments_root, family, examples, crops, args.pretrained, device)
        for k in sorted(examples["set"].unique()):
            plot_box_family(results, family, int(k), out / f"{family}_set{k}.png")
        table += [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        photos = {r["photo"]: r for r in results}.values()
        print(f"{family}: crops {sum(r['correct'] for r in results)}/{len(results)} correct, "
              f"photos {sum(r['photo_correct'] for r in photos)}/{len(photos)} correct, "
              f"{sum(r['matches_prediction_file'] for r in results)}/{len(results)} crops and "
              f"{sum(r['matches_photo_prediction_file'] for r in photos)}/{len(photos)} photos match prediction files",
              flush=True)
    pd.DataFrame(table).to_csv(out / "gradcam_results.csv", index=False)
    print(f"Wrote examples.csv, gradcam_results.csv and figures to {out}")
    return 0


def main_all_test(args: argparse.Namespace, device: torch.device) -> int:
    """Every test photo of --classes: the full image (main run) and each of its per-box crops (this run)."""
    out = args.experiments_root / "gradcam_all"
    out.mkdir(parents=True, exist_ok=True)
    photos = all_test_images(args.roi_manifest, args.classes)
    photos.to_csv(out / "examples.csv", index=False)
    crops = pd.read_csv(args.box_manifest, dtype=str, keep_default_na=False)
    crops = crops[crops["split"].str.strip().str.lower() == "test"]
    crop_rows = crops.set_index("roi_relative_path")
    table = []
    for family in args.families:
        full = explain(args.main_root, family, "full", photos, args.pretrained, device)
        for r, photo in zip(full, photos[MANIFEST_PHOTO_COLUMN]):
            r["photo"] = photo
        boxes = explain_box(args.experiments_root, family, photos, crops, args.pretrained, device)
        size_of = {r["photo"]: r["_image"].shape[0] for r in full}
        cam_of = {r["photo"]: r["_cam"] for r in full}
        for r in boxes:
            crop = crop_rows.loc[r["image"]]
            r["crop_index"] = crop["crop_index"]
            r["_full_box"] = wound_box(crop.to_dict(), size_of[r["photo"]])
            r["full_cam_in_box"], r["box_share"] = mass_in_box(cam_of[r["photo"]], r["_full_box"])
        for label in sorted(photos[LABEL_COLUMN].unique()):
            plot_class_crops(full, boxes, family, label, out / f"{family}_{label.replace(' ', '_')}.png")
        table += [{k: v for k, v in r.items() if not k.startswith("_")} for r in full + boxes]
        print(f"{family}: full {sum(r['correct'] for r in full)}/{len(full)} photos correct, "
              f"per-box {sum(r['correct'] for r in boxes)}/{len(boxes)} crops correct, "
              f"{sum(r['matches_prediction_file'] for r in full + boxes)}/{len(full) + len(boxes)} match "
              f"prediction files, full-image CAM in crop box {mean_of(boxes, 'full_cam_in_box'):.2f} "
              f"(box area {mean_of(boxes, 'box_share'):.2f})", flush=True)
    pd.DataFrame(table).to_csv(out / "gradcam_results.csv", index=False)
    print(f"Wrote examples.csv, gradcam_results.csv and figures to {out}")
    return 0


def main_auto(args: argparse.Namespace, device: torch.device) -> int:
    """T19: the example photos' auto-box crops, explained by the existing union or per-box models."""
    out = args.experiments_root / f"gradcam_auto_{args.auto_arm}"
    out.mkdir(parents=True, exist_ok=True)
    examples = pd.read_csv(args.examples, dtype=str, keep_default_na=False)
    examples.to_csv(out / "examples.csv", index=False)
    table = []
    for family in args.families:
        if args.box_manifest is not None:
            crops = pd.read_csv(args.box_manifest, dtype=str, keep_default_na=False)
            crops = crops[crops["split"].str.strip().str.lower() == "test"]
            results = explain_box(args.experiments_root, family, examples, crops, args.pretrained, device, args.auto_root)
            for k in sorted(examples["set"].unique()):
                plot_box_family(results, family, int(k), out / f"{family}_set{k}.png")
        else:
            auto = pd.read_csv(args.auto_manifest, dtype=str, keep_default_na=False).set_index(MANIFEST_PHOTO_COLUMN)
            auto_examples = examples.assign(roi_relative_path=auto.loc[examples[MANIFEST_PHOTO_COLUMN], "roi_relative_path"].to_numpy())
            manual = explain(args.experiments_root, family, "roi", examples, args.pretrained, device)
            automatic = explain(args.experiments_root, family, "roi", auto_examples, args.pretrained, device, args.auto_root)
            for r in automatic:
                r["input_path"] = "auto roi"
            results = manual + automatic
            for k in sorted(examples["set"].unique()):
                by_key = {(r["class"], r["input_path"]): r for r in results if int(r["set"]) == int(k)}
                rows = [(label, {"manual roi": by_key.get((label, "roi")), "auto roi": by_key.get((label, "auto roi"))})
                        for label in sorted({r["class"] for r in results})]
                plot_rows(rows, f"{family} Grad-CAM, manual vs auto ({args.auto_arm}) union crop, test set {k}",
                          out / f"{family}_set{k}.png")
        table += [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        print(f"{family}: {sum(r['correct'] for r in results)}/{len(results)} correct", flush=True)
    pd.DataFrame(table).to_csv(out / "gradcam_results.csv", index=False)
    print(f"Wrote examples.csv, gradcam_results.csv and figures to {out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    if args.auto_arm is not None:
        return main_auto(args, device)
    if args.all_test:
        return main_all_test(args, device)
    if args.box_manifest is not None:
        return main_box(args, device)
    out = args.experiments_root / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
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
