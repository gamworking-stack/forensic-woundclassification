"""Evaluation figures: confusion matrix and one-vs-rest ROC curves, fixed class order."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical colours by class position, so a class keeps its colour in every scenario's figure.
# Dash patterns are a second cue for greyscale print and colour-blind readers.
CLASS_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CLASS_DASHES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (8, 2)), (0, (4, 1, 1, 1, 1, 1)), (0, (1, 3)))
INK = "#1a1a19"
MUTED = "#8a8a85"


def plot_confusion_matrix(table: pd.DataFrame, title: str, path: Path) -> None:
    counts = table.to_numpy()
    row_totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, row_totals, out=np.zeros(counts.shape, dtype=float), where=row_totals > 0)

    size = len(table)
    fig, ax = plt.subplots(figsize=(1.1 * size + 2.5, 1.0 * size + 2.0))
    ax.imshow(share, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(size):
        for j in range(size):
            colour = "white" if share[i, j] > 0.6 else "black"
            ax.text(j, i, f"{counts[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color=colour)
    ax.set_xticks(range(size), table.columns, rotation=40, ha="right")
    ax.set_yticks(range(size), table.index)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_curves(curves: pd.DataFrame, classes: list, title: str, path: Path) -> None:
    """One-vs-rest ROC curve per class, the macro average and the chance line; AUCs in the legend."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="chance (AUC = 0.500)")
    for k, name in enumerate(classes):
        curve = curves[curves["curve"] == name]
        if curve.empty:
            continue
        ax.plot(curve["fpr"], curve["tpr"], color=CLASS_COLOURS[k % len(CLASS_COLOURS)],
                linestyle=CLASS_DASHES[k % len(CLASS_DASHES)], linewidth=1.5,
                label=f"{name} (AUC = {curve['auc'].iloc[0]:.3f})")
    macro = curves[curves["curve"] == "macro average"]
    ax.plot(macro["fpr"], macro["tpr"], color=INK, linewidth=2.5,
            label=f"macro average (AUC = {macro['auc'].iloc[0]:.3f})")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.grid(color="#e5e4df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


FAMILY_LABELS = {
    "efficientnet_b0": "EfficientNet-B0", "resnet152": "ResNet152", "mobilenet_v3_small": "MobileNetV3-Small",
    "googlenet": "GoogLeNet", "vgg16": "VGG16", "vit_b16": "ViT-B/16", "cnn_svm": "CNN-SVM",
    "alexnet_ensemble": "AlexNet ensemble",
}


def plot_full_vs_roi(rows: pd.DataFrame, pooled: str, title: str, path: Path) -> None:
    """Forest plot: ROI - full difference with its 95% interval per family, pooled row last."""
    families = [f for f in rows["family"] if f != pooled]
    order = families + [pooled]
    data = rows.set_index("family").loc[order]
    y = np.arange(len(order))[::-1].astype(float)
    y[-1] -= 0.5  # gap above the pooled row

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(order) + 1.6))
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    for yi, (family, r) in zip(y, data.iterrows()):
        is_pooled = family == pooled
        ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=INK, linewidth=2.5 if is_pooled else 1.5,
                solid_capstyle="round", zorder=2)
        ax.plot(r["difference"], yi, marker="D" if is_pooled else "o", markersize=8 if is_pooled else 6.5,
                color=INK, markeredgecolor="white", markeredgewidth=1, zorder=3)
        ax.annotate(f"{r['difference']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", xy=(1.02, yi),
                    xycoords=("axes fraction", "data"), va="center", fontsize=8, color=INK)

    reach = max(abs(data["ci_low"]).max(), abs(data["ci_high"]).max(), 0.01) * 1.1
    ax.set_xlim(-reach, reach)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_yticks(y, [FAMILY_LABELS.get(f, f) for f in order])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("ROI − full (95% paired case-bootstrap interval)")
    ax.text(0.0, -0.17, "← full image better", transform=ax.transAxes, ha="left", fontsize=8, color=MUTED)
    ax.text(1.0, -0.17, "ROI better →", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    ax.set_title(title)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
