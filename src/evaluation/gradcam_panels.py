#!/usr/bin/env python3
"""Grad-CAM panels for chosen test photos, one image file per family and input, for slides.

Each photo is explained by each requested family on the full image (main run, S01–S15) and on the
photo's single wound crop (per-crop run, S25–S32), with the same checkpoints, preprocessing and
Grad-CAM as ``gradcam.py``. Every panel is saved on its own (``<case>_<family>_<input>.png``: the
input overlaid with the map, dashed wound box on the full image), and ``panels.csv`` records the
prediction, whether it is right, whether it matches the reported prediction file and, for the full
image, the share of the map inside the wound box.

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
from src.evaluation.gradcam import FAMILIES, checkpoint_folder, grad_cam, load_model, mass_in_box, wound_box  # noqa: E402
from src.training.common import resolve_device  # noqa: E402
from src.training.data import GROUP_COLUMN, LABEL_COLUMN, load_image  # noqa: E402
from src.training.photo import MANIFEST_PHOTO_COLUMN  # noqa: E402
from src.training.transforms import IMAGENET_MEAN, IMAGENET_STD, PadToSquare  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM panels: full image and wound crop for chosen test photos.")
    parser.add_argument("--main-root", type=Path, required=True, help="Main run (full-image models).")
    parser.add_argument("--crop-root", type=Path, required=True, help="Per-crop run (wound-crop models).")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--photos", nargs="+", required=True, help="Test photos (manifest relative_path), one crop each.")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def panel(experiments: Path, input_path: str, family: str, row: Dict, device) -> Dict:
    """Grad-CAM of one family on one input of one photo, as ``gradcam.explain`` computes it."""
    # Checkpoint, run config, model and the training preprocessing of the scenario
    folder = checkpoint_folder(experiments, input_path, family)
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    classes, size, root = config["classes"], int(config["image_size"]), Path(config["image_root"])
    model = load_model(folder, family, len(classes), True, device)
    show = T.Compose([PadToSquare(), T.Resize((size, size))])
    to_input = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    # The source photo for the full image, the crop file for the wound crop
    path = find_source_image(root, row) if input_path == "full" else root / row["roi_relative_path"]
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"{input_path} image of {row[MANIFEST_PHOTO_COLUMN]} not found under {root}")
    image = show(load_image(Path(path), 2 * size if config.get("cache", True) else None))
    cam, probs = grad_cam(model, family, to_input(image).unsqueeze(0).to(device))
    # Prediction, its agreement with the reported prediction file, and the map's share in the wound box
    predicted = classes[int(probs.argmax())]
    reported = pd.read_csv(folder / "test_predictions.csv", dtype=str).set_index("relative_path")["predicted_class"]
    key = Path(path).relative_to(root).as_posix()
    box = wound_box(row, size) if input_path == "full" else None
    return {"case": row[GROUP_COLUMN], "photo": row[MANIFEST_PHOTO_COLUMN], "family": family, "input_path": input_path,
            "class": row[LABEL_COLUMN], "predicted_class": predicted, "probability": float(probs.max()),
            "correct": predicted == row[LABEL_COLUMN], "matches_prediction_file": reported.get(key) == predicted,
            "cam_in_box": mass_in_box(cam, box)[0] if box else np.nan,
            "_image": np.asarray(image), "_cam": cam, "_box": box}


def save_panel(r: Dict, path: Path) -> None:
    """The input overlaid with the map (inferno, as gradcam.py), dashed wound box on the full image."""
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(r["_image"])
    ax.imshow(r["_cam"], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    if r["_box"] is not None:
        x1, y1, x2, y2 = r["_box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1.5, linestyle="--"))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    crops = pd.read_csv(args.box_manifest, dtype=str, keep_default_na=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for photo in args.photos:
        # The photo's single crop: its manifest row serves both inputs (photo fields and crop box)
        of_photo = crops[crops[MANIFEST_PHOTO_COLUMN] == photo]
        if len(of_photo) != 1:
            raise ValueError(f"{photo!r} has {len(of_photo)} crops; choose photos with exactly one")
        row = of_photo.iloc[0].to_dict()
        for family in args.families:
            for experiments, input_path in ((args.main_root, "full"), (args.crop_root, "crop")):
                r = panel(experiments, input_path, family, row, device)
                save_panel(r, args.output_dir / f"case{r['case']}_{family}_{input_path}.png")
                results.append(r)
                print(f"case {r['case']} {family} {input_path}: {'right' if r['correct'] else 'wrong'} "
                      f"({r['predicted_class']} {r['probability']:.2f}); matches prediction file: {r['matches_prediction_file']}")
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results]).to_csv(
        args.output_dir / "panels.csv", index=False)
    print(f"Wrote {len(results)} panels and panels.csv to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
