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
