#!/usr/bin/env python3
"""Manuscript figures (T27): Figure 1 study pipeline (full image and per-crop ROI) and models, Figure 2 photograph flow.

Figure 2's counts are read from the sheet and the committed T23 manifests (after T26), so the figure
always matches Table 1: photographs and patients in the sheet, each exclusion reason, and photographs,
patients and wound crops per split. Only counts are drawn; no photograph or identifier. Each figure is
written as PNG and TIFF (600 dpi) and PDF to ``--output-dir`` (outside the repository).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.data.prepare_relabel import PATIENT_PLACEHOLDERS, read_sheet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
INK, MUTED, LINE = "#1f2328", "#57606a", "#8c959f"
FILL = {"input": "#eef2f7", "model": "#e8f1ec", "eval": "#f6f0e6", "plain": "#ffffff", "exclude": "#f7eded"}
ACCENT = {"full": "#3b6ea5", "union": "#2e8b57", "box": "#c0782f"}
SPLITS = (("train", "Training"), ("validation", "Validation"), ("test", "Test"))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the manuscript figures.")
    parser.add_argument("--sheet", type=Path, required=True,
                        help="The researcher's sheet as CSV (e.g. ../thesis_data_27_oct_corrected.csv).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    return parser.parse_args(argv)


def setup_fonts() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})


def box(ax, x, y, w, h, text, fill="plain", edge=LINE, size=7, weight="normal", align="center", lw=0.8):
    """A rounded box centred on (x, y) in axes units, with its text."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=FILL.get(fill, fill), edgecolor=edge, linewidth=lw))
    tx = x if align == "center" else x - w / 2 + 0.012
    ax.text(tx, y, text, ha=align, va="center", fontsize=size, color=INK, fontweight=weight, linespacing=1.35)


def titled(ax, x, y, w, h, title, body, fill="plain", edge=LINE, colour=INK, size=6.5, lw=0.8):
    """A rounded box with a bold title line above its body, the pair centred vertically."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=FILL.get(fill, fill), edgecolor=edge, linewidth=lw))
    # Line height in axes units: points -> inches -> share of the figure height
    line = size * 1.35 / 72 / ax.figure.get_figheight()
    lines = body.count("\n") + 1
    top = y + (lines + 1) * line / 2 + line * 0.15
    ax.text(x, top - line / 2, title, ha="center", va="center", fontsize=size + 0.4, fontweight="bold", color=colour)
    ax.text(x, top - line * 1.3, body, ha="center", va="top", fontsize=size, color=INK, linespacing=1.35)


def arrow(ax, start, end, colour=LINE, lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=7, color=colour, linewidth=lw,
                                 shrinkA=0, shrinkB=0))


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf", "tiff"):
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(out / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03, **kwargs)
    plt.close(fig)


def photo_icon(ax, x, y, w, h, boxes, crop=None, colour=INK):
    """A schematic photograph with wound boxes; ``crop`` outlines a crop region (axes units)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#d9dee4", edgecolor=MUTED, linewidth=0.6))
    for bx, by, bw, bh in boxes:
        ax.add_patch(Rectangle((x + bx * w, y + by * h), bw * w, bh * h, facecolor="#b5838d", edgecolor="none"))
    if crop is not None:
        for cx, cy, cw, ch in crop:
            ax.add_patch(Rectangle((x + cx * w, y + cy * h), cw * w, ch * h, facecolor="none", edgecolor=colour,
                                   linewidth=0.9, linestyle="--"))


def figure_pipeline(out: Path) -> None:
    """Figure 1: full image and per-crop ROI (Experiment 3), shared preprocessing, eight classifiers, evaluation."""
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    wounds = [(0.18, 0.55, 0.16, 0.2), (0.55, 0.3, 0.2, 0.16)]
    crops = [(0.15, 0.52, 0.22, 0.26), (0.52, 0.27, 0.26, 0.22)]
    rows = {"full": 0.72, "box": 0.33}

    # Column a: the photograph and its manual boxes
    ax.text(0.085, 0.965, "a  Input", fontsize=7.5, fontweight="bold", color=INK, ha="center")
    photo_icon(ax, 0.03, 0.48, 0.11, 0.17, wounds)
    ax.text(0.085, 0.44, "Forensic wound\nphotograph", ha="center", va="top", fontsize=6.5, color=INK)
    ax.text(0.085, 0.32, "Wound regions boxed\nmanually (CVAT)", ha="center", va="top", fontsize=6.5, color=MUTED)

    # Column b: the two inputs
    ax.text(0.30, 0.965, "b  Classifier input", fontsize=7.5, fontweight="bold", color=INK, ha="center")
    photo_icon(ax, 0.205, rows["full"] - 0.075, 0.075, 0.15, wounds)
    ax.text(0.29, rows["full"] + 0.035, "Full image", fontsize=7, fontweight="bold", color=ACCENT["full"], va="center")
    ax.text(0.29, rows["full"] - 0.035, "whole photograph", fontsize=6, color=MUTED, va="center")
    photo_icon(ax, 0.205, rows["box"] + 0.02, 0.075, 0.15, wounds, crops, ACCENT["box"])
    # The crops, each cut out on its own
    for i, (bx, by, bw, bh) in enumerate(wounds):
        x0 = 0.215 + i * 0.04
        ax.add_patch(Rectangle((x0, rows["box"] - 0.12), 0.03, 0.09, facecolor="#d9dee4",
                               edgecolor=ACCENT["box"], linewidth=0.9, linestyle="--"))
        ax.add_patch(Rectangle((x0 + 0.008, rows["box"] - 0.097), 0.014, 0.045, facecolor="#b5838d", edgecolor="none"))
    arrow(ax, (0.2425, rows["box"] + 0.015), (0.2425, rows["box"] - 0.025), colour=ACCENT["box"], lw=0.7)
    ax.text(0.29, rows["box"] + 0.135, "Per-crop ROI", fontsize=7, fontweight="bold", color=ACCENT["box"], va="center")
    ax.text(0.29, rows["box"] + 0.06, "one crop per wound box\n(+8% margin), each\nclassified on its own",
            fontsize=6, color=MUTED, va="center", linespacing=1.3)
    for key, y in rows.items():
        arrow(ax, (0.145, 0.565), (0.2, y), colour=LINE, lw=0.7)
        arrow(ax, (0.415, y), (0.445, y), colour=ACCENT[key], lw=0.9)

    # Column c: preprocessing shared by both inputs
    ax.text(0.515, 0.965, "c  Preprocessing", fontsize=7.5, fontweight="bold", color=INK, ha="center")
    ax.add_patch(FancyBboxPatch((0.445, 0.12), 0.14, 0.78, boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=FILL["input"], edgecolor=LINE, linewidth=0.8))
    ax.text(0.515, 0.51, "Pad to square\n\nResize to\n384 × 384 px\n(ViT-B/16:\n224 × 224 px)\n\nTraining only:\nflips, rotation\n±15°, random\nresized crop,\nbrightness and\ncontrast, noise",
            ha="center", va="center", fontsize=6.3, color=INK, linespacing=1.3)
    arrow(ax, (0.585, 0.51), (0.61, 0.51))

    # Column d: eight classifiers in three groups, trained separately for each input
    ax.text(0.72, 0.965, "d  Classifiers (8)", fontsize=7.5, fontweight="bold", color=INK, ha="center")
    titled(ax, 0.72, 0.665, 0.21, 0.40, "Fine-tuned",
           "ImageNet-pretrained,\nall layers trained\n\nEfficientNet-B0\nResNet152\nMobileNetV3-Small\nGoogLeNet\nVGG16\nViT-B/16",
           fill="model", size=6.5)
    titled(ax, 0.72, 0.355, 0.21, 0.17, "CNN-SVM",
           "frozen VGG16 + ResNet152 +\nEfficientNet-B0 features → SVM", fill="model", size=6.3)
    titled(ax, 0.72, 0.18, 0.21, 0.135, "AlexNet ensemble", "3 seeds, soft voting", fill="model", size=6.3)
    ax.text(0.72, 0.075, "Each trained separately per input;\noutput: probabilities of 7 wound classes",
            ha="center", va="center", fontsize=6.0, color=MUTED, linespacing=1.3)

    # Column e: one prediction per photograph or per crop, then evaluation
    ax.text(0.925, 0.965, "e  Output", fontsize=7.5, fontweight="bold", color=INK, ha="center")
    titled(ax, 0.92, rows["full"], 0.115, 0.17, "Full image", "one prediction\nper photograph",
           edge=ACCENT["full"], colour=ACCENT["full"], size=6.2, lw=0.9)
    titled(ax, 0.92, rows["box"] + 0.12, 0.115, 0.19, "Experiment 3", "per-crop ROI:\none prediction\nper wound crop",
           edge=ACCENT["box"], colour=ACCENT["box"], size=6.2, lw=0.9)
    arrow(ax, (0.825, rows["full"]), (0.8625, rows["full"]), colour=ACCENT["full"])
    arrow(ax, (0.825, rows["box"] + 0.12), (0.8625, rows["box"] + 0.12), colour=ACCENT["box"])
    # Both outputs are evaluated: the full-image path runs down the right edge, past Experiment 3
    ax.plot([0.9775, 0.988, 0.988], [rows["full"], rows["full"], 0.30], color=ACCENT["full"], linewidth=0.9)
    arrow(ax, (0.988, 0.30), (0.988, 0.27), colour=ACCENT["full"])
    arrow(ax, (0.92, rows["box"] + 0.025), (0.92, 0.27), colour=ACCENT["box"])
    titled(ax, 0.925, 0.16, 0.135, 0.22, "Evaluation",
           "held-out test set\n(patient-level split)\n5-fold cross-validation\naccuracy, balanced\naccuracy, macro-F1,\nmacro AUC; 95% CI",
           fill="eval", size=5.8)
    save(fig, out, "figure1_pipeline")


def flow_counts(sheet_path: Path, data_dir: Path) -> Dict:
    """Photographs and patients at each step, from the sheet and the committed manifests."""
    sheet, _ = read_sheet(sheet_path)
    known = ~sheet["patient_id"].str.strip().str.lower().isin(PATIENT_PLACEHOLDERS)
    patients = sheet["patient_id"].where(known, "case:" + sheet["pseudonimized_id"])
    excluded = pd.read_csv(data_dir / "excluded_photos.csv", dtype=str, keep_default_na=False)
    included = pd.read_csv(data_dir / "splits/patient_cases.csv", dtype=str, keep_default_na=False)
    crops = pd.read_csv(data_dir / "roi_box/roi_manifest.csv", dtype=str, keep_default_na=False)
    if len(sheet) != len(excluded) + len(included):
        raise ValueError(f"Sheet photos {len(sheet)} != excluded {len(excluded)} + included {len(included)}")
    development = included[included["split"] != "test"]
    return {
        "sheet_photos": len(sheet), "sheet_patients": patients.nunique(),
        "reasons": excluded["reason"].value_counts().to_dict(), "excluded": len(excluded),
        "photos": len(included), "patients": included["patient_case"].nunique(), "crops": len(crops),
        "patients_lost": patients.nunique() - included["patient_case"].nunique(),
        "split": {s: (int((included["split"] == s).sum()), included.loc[included["split"] == s, "patient_case"].nunique(),
                      int((crops["split"] == s).sum())) for s, _ in SPLITS},
        "development": (len(development), development["patient_case"].nunique()),
    }


def figure_flow(counts: Dict, out: Path) -> None:
    """Figure 2: photographs from the case record to the training, validation and test sets."""
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    r = counts["reasons"]
    box(ax, 0.36, 0.92, 0.56, 0.1,
        f"Forensic wound photographs in the case record\nn = {counts['sheet_photos']:,} photographs "
        f"({counts['sheet_patients']:,} patients)", size=7.2)
    ax.add_patch(FancyBboxPatch((0.615, 0.655), 0.375, 0.16, boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=FILL["exclude"], edgecolor=LINE, linewidth=0.8))
    ax.text(0.63, 0.79, f"Excluded: n = {counts['excluded']} photographs", fontsize=6.8, fontweight="bold",
            color=INK, va="center")
    ax.text(0.63, 0.765,
            f"• Did not meet inclusion criteria on review: {r.get('inclusion_false', 0)}\n"
            f"• Imprint (outside the 7 study classes): {r.get('label_outside_7_classes', 0)}\n"
            f"• No wound bounding box annotated: {r.get('no_annotation', 0)}\n"
            f"{counts['patients_lost']} patients had no photograph left",
            fontsize=6.3, color=INK, va="top", linespacing=1.4)
    arrow(ax, (0.36, 0.87), (0.36, 0.60))
    arrow(ax, (0.36, 0.735), (0.61, 0.735), style="-")
    arrow(ax, (0.60, 0.735), (0.61, 0.735))
    box(ax, 0.36, 0.545, 0.56, 0.11,
        f"Included: n = {counts['photos']:,} photographs ({counts['patients']} patients), 7 wound classes\n"
        f"{counts['crops']:,} annotated wound regions (bounding boxes)", fill="input", size=7.2, weight="normal")
    ax.text(0.375, 0.457, "Patient-level split (70 / 15 / 15)", ha="left", va="center", fontsize=6.5, color=MUTED,
            style="italic")
    xs = (0.17, 0.50, 0.83)
    arrow(ax, (0.36, 0.49), (0.36, 0.425), style="-")
    ax.plot([xs[0], xs[2]], [0.425, 0.425], color=LINE, linewidth=0.9)
    for (key, label), x in zip(SPLITS, xs):
        photos, patients, crops = counts["split"][key]
        arrow(ax, (x, 0.425), (x, 0.375))
        titled(ax, x, 0.30, 0.29, 0.15, f"{label} set", f"{photos} photographs\n{patients} patients\n{crops:,} wound crops",
               fill="eval" if key == "test" else "plain", size=6.8)
    dev_photos, dev_patients = counts["development"]
    arrow(ax, (xs[0], 0.225), (xs[0], 0.165), style="-")
    arrow(ax, (xs[1], 0.225), (xs[1], 0.165), style="-")
    ax.plot([xs[0], xs[1]], [0.165, 0.165], color=LINE, linewidth=0.9)
    arrow(ax, ((xs[0] + xs[1]) / 2, 0.165), ((xs[0] + xs[1]) / 2, 0.13))
    titled(ax, (xs[0] + xs[1]) / 2, 0.08, 0.40, 0.09, "5-fold cross-validation",
           f"{dev_photos} photographs, {dev_patients} patients; no patient in two folds", size=6.5)
    titled(ax, xs[2], 0.08, 0.29, 0.09, "Final evaluation", "held out until all models\nwere chosen; used once", fill="eval", size=6.5)
    arrow(ax, (xs[2], 0.225), (xs[2], 0.125))
    save(fig, out, "figure2_flow")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; figures go to results_figures")
    out.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    figure_pipeline(out)
    counts = flow_counts(args.sheet, args.data_dir)
    figure_flow(counts, out)
    print(f"Figure 2 counts: {counts}")
    print(f"Wrote figure1_pipeline and figure2_flow (.png, .pdf, .tiff) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
