#!/usr/bin/env python3
"""Test confusion matrices per subgroup (age, race/ethnicity, skin tone, setting, body region), T32.

The subgroups, photo matching and minimum size are those of ``subgroup.py`` (T24, manuscript Table 3).
For each attribute and input path (full image S01–S15, per crop S25–S32) one figure: a row per
subgroup, a column per chosen family, each cell the count of test images, coloured by its share of
the true class. ``subgroup_confusion.csv`` holds the same counts. No per-image rows are written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.evaluation.evaluate import CROP_PATH, FAMILY_ORDER  # noqa: E402
from src.evaluation.plots import FAMILY_LABELS, INK  # noqa: E402
from src.evaluation.subgroup import (ATTRIBUTES, PHOTO, REPO_ROOT, labelled_predictions, photo_keys,  # noqa: E402
                                     prediction_file, scored_levels, test_photos)

DEFAULT_FAMILIES = ("resnet152", "efficientnet_b0", "vgg16", "mobilenet_v3_small")
DEFAULT_ATTRIBUTES = ("age_group", "race_ethnicity", "skin_tone", "setting", "body_region")
ATTRIBUTE_LABELS = {"age_group": "Age group", "race_ethnicity": "Race/ethnicity", "skin_tone": "Monk Skin Tone",
                    "setting": "Setting", "body_region": "Body region", "tattoo": "Tattoo"}
LEVEL_LABELS = {"18-39": "18–39 years", "40-59": "40–59 years", "60+": "≥60 years", "MST 1-3": "MST 1–3",
                "MST 5-10": "MST 5–10", "autopsy": "Deceased (autopsy)"}
PATH_LABELS = {"full": "Full image", CROP_PATH: "Per-crop ROI"}
CLASS_SHORT = {"contusion": "CON", "cut wound": "CUT", "graze wound": "GRZ", "gunshot wound": "GSW",
               "laceration": "LAC", "penetration wound": "PEN", "scratch": "SCR"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test confusion matrices per subgroup and family.")
    parser.add_argument("--sheet", type=Path, required=True, help="The researcher's sheet as CSV (as subgroup.py).")
    parser.add_argument("--main-root", type=Path, required=True, help="Main run (full/<family>/test_predictions.csv).")
    parser.add_argument("--crop-root", type=Path, required=True, help="Per-crop run (crop/<family>/test_predictions.csv).")
    parser.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=list(DEFAULT_FAMILIES))
    parser.add_argument("--attributes", nargs="+", choices=ATTRIBUTES, default=list(DEFAULT_ATTRIBUTES))
    parser.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--split-manifest", type=Path, default=DATA_DIR / "splits/case_level_split_manifest.csv")
    parser.add_argument("--roi-manifest", type=Path, default=DATA_DIR / "roi/roi_manifest.csv")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--min-photos", type=int, default=10, help="Smallest subgroup shown (as subgroup.py).")
    return parser.parse_args(argv)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def plot_grid(matrices: dict, levels: List[str], families: List[str], classes: List[str], path: Path) -> None:
    """A row per subgroup, a column per family: counts, coloured by the share of the true class.

    Class labels sit on the outer panels only; each panel is titled with its images and accuracy.
    """
    short = [CLASS_SHORT.get(c, c) for c in classes]
    k = len(classes)
    fig, axes = plt.subplots(len(levels), len(families), figsize=(2.2 * len(families) + 1.4, 2.25 * len(levels) + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.08, top=0.92, wspace=0.08, hspace=0.28)
    for i, level in enumerate(levels):
        for j, family in enumerate(families):
            ax, counts = axes[i, j], matrices[(level, family)]
            totals = counts.sum(axis=1, keepdims=True)
            share = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
            image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for r in range(k):
                for c in range(k):
                    if counts[r, c]:
                        ax.text(c, r, str(counts[r, c]), ha="center", va="center", fontsize=6.5,
                                color="white" if share[r, c] > 0.6 else INK)
            ax.set_xticks(range(k), short, rotation=90, fontsize=6.5)
            ax.set_yticks(range(k), short, fontsize=6.5)
            ax.tick_params(length=0, pad=1.5)
            n, correct = int(counts.sum()), int(np.trace(counts))
            ax.set_title(f"n = {n}, accuracy {correct / n:.2f}" if n else "n = 0", fontsize=7, color=INK, pad=3)
            if i == 0:
                ax.annotate(FAMILY_LABELS.get(family, family), (0.5, 1.22), xycoords="axes fraction",
                            ha="center", fontsize=9, fontweight="bold", color=INK)
            if j == 0:
                ax.set_ylabel("True class", fontsize=7, color=INK)
                ax.annotate(LEVEL_LABELS.get(level, level[:1].upper() + level[1:]), (-0.42, 0.5),
                            xycoords="axes fraction", rotation=90, ha="center", va="center", fontsize=9,
                            fontweight="bold", color=INK)
            if i == len(levels) - 1:
                ax.set_xlabel("Predicted class", fontsize=7, color=INK)
    bar = fig.colorbar(image, cax=fig.add_axes([0.9, 0.3, 0.015, 0.4]))
    bar.set_label("Share of the true class", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; subgroup results stay outside it")
    out.mkdir(parents=True, exist_ok=True)

    photos = test_photos(args.sheet, args.split_manifest)
    keys = photo_keys(photos, args.roi_manifest, args.box_manifest)
    roots = {"full": args.main_root, CROP_PATH: args.crop_root}
    by_path = {p: {f: labelled_predictions(prediction_file(root, p, f), p, keys[p], photos) for f in args.families}
               for p, root in roots.items()}
    # Subgroups as subgroup.py scores them: over the photos the full-image models were scored on
    photos = photos[photos["relative_path"].isin(by_path["full"][args.families[0]].frame[PHOTO])]
    levels = scored_levels(photos, args.min_photos)

    rows = []
    for input_path, predictions in by_path.items():
        classes = predictions[args.families[0]].classes
        for attribute in args.attributes:
            matrices = {}
            for family, p in predictions.items():
                if p.classes != classes:
                    raise ValueError(f"{family} {input_path} has another class order than {args.families[0]}")
                for level in levels[attribute]:
                    mask = (p.frame[attribute] == level).to_numpy()
                    matrices[(level, family)] = confusion(p.y_true[mask], p.y_pred[mask], len(classes))
                    for t, true_class in enumerate(classes):
                        for q, predicted in enumerate(classes):
                            rows.append({"input_path": input_path, "attribute": attribute, "subgroup": level,
                                         "family": family, "true_class": true_class, "predicted_class": predicted,
                                         "count": int(matrices[(level, family)][t, q])})
            name = f"confusion_{input_path}_{attribute}.png"
            plot_grid(matrices, levels[attribute], args.families, classes, out / name)
            print(f"{input_path} {attribute}: {', '.join(levels[attribute])} -> {name}", flush=True)
    pd.DataFrame(rows).to_csv(out / "subgroup_confusion.csv", index=False)
    print(f"Wrote subgroup_confusion.csv and {len(args.attributes) * len(roots)} figures to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
