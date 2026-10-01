"""Confusion matrix figure: counts with row percentages, fixed class order."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


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
