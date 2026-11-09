#!/usr/bin/env python3
"""Grad-CAM panels for chosen test photos, one image file per family and input, for slides.

Each photo is explained by each requested family on the full image (main run, S01–S15) and on the
photo's single wound crop (per-crop run, S25–S32), with the same checkpoints, preprocessing and
Grad-CAM as ``gradcam.py``. Every panel is saved on its own (``<case>_<family>_<input>.png``: the
input overlaid with the map, dashed wound box on the full image), and ``panels.csv`` records the
prediction, whether it is right, whether it matches the reported prediction file and, for the full
image, the share of the map inside the wound box.

With ``--all-test`` every test photo is explained instead, with every one of its crops (photos with
several wounds included): panels go to ``<family>/<class>/case<case>_<photo>_full.png`` (all the
photo's wound boxes drawn) and ``..._crop<k>.png``, and ``<family>/<class>_p<page>.png`` shows each
crop next to its full image with that crop's box, 20 crops per page. Each model is loaded once.
``--photo-list`` does the same for the photos listed in a CSV (``relative_path``), from any split: panels go
to ``<family>/<split>/<class>/`` and the overview pages to ``<family>/<split>_p<page>.png``. Training
photos have no prediction file, so their ``matches_prediction_file`` is left empty.

The panels are overlays on patient photographs: they stay on Raptor; any copy used outside it must be
de-identified and allowed by the ethics approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from torchvision import transforms as T  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_roi_dataset import find_source_image  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402
from src.evaluation.gradcam import FAMILIES, checkpoint_folder, grad_cam, load_model, plot_rows, wound_box  # noqa: E402
from src.training.common import resolve_device  # noqa: E402
from src.training.data import GROUP_COLUMN, LABEL_COLUMN, load_image  # noqa: E402
from src.training.photo import MANIFEST_PHOTO_COLUMN  # noqa: E402
from src.training.transforms import IMAGENET_MEAN, IMAGENET_STD, PadToSquare  # noqa: E402

PAGE_ROWS = 20  # crops per overview page (--all-test)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM panels: full image and wound crop for chosen test photos.")
    parser.add_argument("--main-root", type=Path, required=True, help="Main run (full-image models).")
    parser.add_argument("--crop-root", type=Path, required=True, help="Per-crop run (wound-crop models).")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--photos", nargs="+", help="Test photos (manifest relative_path), one crop each.")
    which.add_argument("--all-test", action="store_true", help="Every test photo and every one of its crops.")
    which.add_argument("--photo-list", type=Path,
                       help="CSV with a relative_path column: every listed photo (any split) and every one of its crops.")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Build backbones without ImageNet weights before loading the checkpoint (tests).")
    return parser.parse_args(argv)


def scenario(experiments: Path, input_path: str, family: str, pretrained: bool, device) -> Dict:
    """Checkpoint, model, preprocessing and reported test predictions of one scenario, loaded once."""
    folder = checkpoint_folder(experiments, input_path, family)
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    size = int(config["image_size"])
    # Reported predictions of the validation and test images (training images have none)
    reported = pd.concat([pd.read_csv(folder / f"{split}_predictions.csv", dtype=str)
                          for split in ("validation", "test") if (folder / f"{split}_predictions.csv").is_file()])
    reported = reported.set_index("relative_path")["predicted_class"]
    return {"family": family, "input_path": input_path, "classes": config["classes"], "size": size,
            "root": Path(config["image_root"]), "cache": config.get("cache", True), "reported": reported,
            "model": load_model(folder, family, len(config["classes"]), pretrained, device),
            "show": T.Compose([PadToSquare(), T.Resize((size, size))]),
            "to_input": T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])}


def mass_in_boxes(cam: np.ndarray, boxes: List) -> float:
    """Share of the map's mass inside any of ``boxes`` (overlaps counted once); NaN without a box or map."""
    if not boxes or cam.sum() <= 0:
        return float("nan")
    inside = np.zeros(cam.shape, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = (max(0, int(round(v))) for v in box)
        inside[y1:y2, x1:x2] = True
    return float(cam[inside].sum() / cam.sum())


def panel(s: Dict, row: Dict, device, boxes: Optional[List[Dict]] = None) -> Dict:
    """Grad-CAM of one scenario on one input of one photo, as ``gradcam.explain`` computes it.

    ``row`` is a crop row of the box manifest; on the full image ``boxes`` are the crop rows whose
    boxes are drawn and scored (default: ``row`` alone).
    """
    # The source photo for the full image, the crop file for the wound crop
    full = s["input_path"] == "full"
    path = find_source_image(s["root"], row) if full else s["root"] / row["roi_relative_path"]
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"{s['input_path']} image of {row[MANIFEST_PHOTO_COLUMN]} not found under {s['root']}")
    image = s["show"](load_image(Path(path), 2 * s["size"] if s["cache"] else None))
    cam, probs = grad_cam(s["model"], s["family"], s["to_input"](image).unsqueeze(0).to(device))
    # Prediction, its agreement with the reported prediction file, and the map's share in the wound boxes
    predicted = s["classes"][int(probs.argmax())]
    key = Path(path).relative_to(s["root"]).as_posix()
    drawn = [b for b in (wound_box(r, s["size"]) for r in (boxes or [row])) if b is not None] if full else []
    known = key in s["reported"].index
    return {"case": row[GROUP_COLUMN], "photo": row[MANIFEST_PHOTO_COLUMN], "split": row.get("split", ""),
            "family": s["family"],
            "input_path": s["input_path"], "crop": "" if full else row["roi_relative_path"],
            "crop_index": "" if full else row.get("crop_index", ""), "class": row[LABEL_COLUMN],
            "predicted_class": predicted, "probability": float(probs.max()), "correct": predicted == row[LABEL_COLUMN],
            "matches_prediction_file": s["reported"].get(key) == predicted if known else np.nan,
            "cam_in_box": mass_in_boxes(cam, drawn) if full else np.nan,
            "_image": np.asarray(image), "_cam": cam, "_boxes": drawn}


def save_panel(r: Dict, path: Path) -> None:
    """The input overlaid with the map (inferno, as gradcam.py), dashed wound boxes on the full image."""
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(r["_image"])
    ax.imshow(r["_cam"], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    for x1, y1, x2, y2 in r["_boxes"]:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1.5, linestyle="--"))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chosen_photos(args: argparse.Namespace, crops: pd.DataFrame, device) -> List[Dict]:
    """--photos: the full image and the single crop of each chosen photo, one file per family and input."""
    results = []
    for family in args.families:
        for experiments, input_path in ((args.main_root, "full"), (args.crop_root, "crop")):
            s = scenario(experiments, input_path, family, args.pretrained, device)
            for photo in args.photos:
                # The photo's single crop: its manifest row serves both inputs (photo fields and crop box)
                of_photo = crops[crops[MANIFEST_PHOTO_COLUMN] == photo]
                if len(of_photo) != 1:
                    raise ValueError(f"{photo!r} has {len(of_photo)} crops; choose photos with exactly one")
                r = panel(s, of_photo.iloc[0].to_dict(), device)
                save_panel(r, args.output_dir / f"case{r['case']}_{family}_{input_path}.png")
                results.append(r)
                print(f"case {r['case']} {family} {input_path}: {'right' if r['correct'] else 'wrong'} "
                      f"({r['predicted_class']} {r['probability']:.2f}); matches prediction file: {r['matches_prediction_file']}")
    return results


def listed_photos(photo_list: Path, crops: pd.DataFrame) -> pd.DataFrame:
    """The crop rows of the photos listed in ``photo_list``; every listed photo must have a crop."""
    listed = set(pd.read_csv(photo_list, dtype=str, keep_default_na=False)["relative_path"])
    missing = sorted(listed - set(crops[MANIFEST_PHOTO_COLUMN]))
    if missing:
        raise ValueError(f"{len(missing)} listed photos have no crop in the box manifest (e.g. {missing[:3]})")
    return crops[crops[MANIFEST_PHOTO_COLUMN].isin(listed)]


def every_crop(args: argparse.Namespace, chosen: pd.DataFrame, device, by_split: bool) -> List[Dict]:
    """Every chosen photo's full image and each of its crops, filed by family and class (and split)."""
    chosen = chosen.sort_values([LABEL_COLUMN, MANIFEST_PHOTO_COLUMN, "roi_relative_path"])
    results = []
    for family in args.families:
        full_s = scenario(args.main_root, "full", family, args.pretrained, device)
        crop_s = scenario(args.crop_root, "crop", family, args.pretrained, device)
        rows: Dict[str, List] = {}
        for photo, of_photo in chosen.groupby(MANIFEST_PHOTO_COLUMN, sort=False):
            # The full image once, with every wound box of the photo; then each crop on its own
            boxes = of_photo.to_dict("records")
            label, split = boxes[0][LABEL_COLUMN], boxes[0]["split"].strip().lower()
            folder = args.output_dir / family / (split if by_split else "") / label.replace(" ", "_")
            folder.mkdir(parents=True, exist_ok=True)
            full = panel(full_s, boxes[0], device, boxes)
            stem = f"case{full['case']}_{Path(photo).stem}"
            save_panel(full, folder / f"{stem}_full.png")
            results.append(full)
            for k, crop_row in enumerate(boxes, start=1):
                r = panel(crop_s, crop_row, device)
                save_panel(r, folder / f"{stem}_crop{crop_row.get('crop_index') or k}.png")
                results.append(r)
                # Overview row: the full image with only this crop's box, next to the crop
                box = wound_box(crop_row, full_s["size"])
                name = f"case {full['case']} {Path(photo).stem} crop {k}/{len(boxes)}"
                rows.setdefault(split if by_split else label, []).append(
                    (f"{name}, {label}" if by_split else name, {"full": {**full, "_box": box}, "crop": r}))
        # Overview pages per class (per split for a photo list), then report this family's counts
        for group, of_group in rows.items():
            for p in range(0, len(of_group), PAGE_ROWS):
                page = p // PAGE_ROWS + 1
                plot_rows(of_group[p:p + PAGE_ROWS], f"{family} Grad-CAM, {group}: full image (box: this crop) "
                          f"and per-crop model, page {page}", args.output_dir / family / f"{group.replace(' ', '_')}_p{page}.png")
        mine = [r for r in results if r["family"] == family]
        for input_path in ("full", "crop"):
            of_path = [r for r in mine if r["input_path"] == input_path]
            known = [r for r in of_path if not pd.isna(r["matches_prediction_file"])]
            print(f"{family} {input_path}: {sum(r['correct'] for r in of_path)}/{len(of_path)} correct, "
                  f"{sum(r['matches_prediction_file'] for r in known)}/{len(known)} match the prediction file", flush=True)
        # Keep only the numbers once the figures are drawn
        for r in mine:
            r.pop("_image"), r.pop("_cam")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    crops = pd.read_csv(args.box_manifest, dtype=str, keep_default_na=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.all_test:
        results = every_crop(args, crops[crops["split"].str.strip().str.lower() == "test"], device, by_split=False)
    elif args.photo_list:
        results = every_crop(args, listed_photos(args.photo_list, crops), device, by_split=True)
    else:
        results = chosen_photos(args, crops, device)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results]).to_csv(
        args.output_dir / "panels.csv", index=False)
    print(f"Wrote {len(results)} panels and panels.csv to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
