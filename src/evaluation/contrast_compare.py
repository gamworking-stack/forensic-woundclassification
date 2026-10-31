#!/usr/bin/env python3
"""T30: more contrast (CLAHE) minus the T23 original, per scenario, on the same test images.

For full image (S01-S15) and per-crop (S25-S32), each family's T30 test predictions are paired with
its T23 predictions; both must hold the same images, cases and true classes. Each resample of the
T15-style paired bootstrap draws one set of cases (sorted cases, ``default_rng(seed)``) and scores both
files of every family on it, so the difference keeps the pairing. Pooled = mean over the 8 families,
its interval from the per-resample means; p-values are two-sided bootstrap p, Holm-adjusted over the
8 families of a metric and input path. Run it on the T25 copies of both runs for patient resampling.

Per wound class (added 2026-10-31) the same resamples give recall, precision, F1 and one-vs-rest AUC
per class, CLAHE minus original (pooled rows Holm-adjusted over the 7 classes), and the images each
model gained (original wrong, CLAHE right) and lost (the reverse).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import CROP_PATH, CROP_SCENARIO_IDS, FAMILY_ORDER, SCENARIO_IDS  # noqa: E402
from src.evaluation.metrics import Predictions, load_predictions, one_vs_rest_auc, summary_metrics  # noqa: E402
from src.evaluation.plots import CLASS_COLOURS, FAMILY_LABELS, INK, MUTED  # noqa: E402
from src.evaluation.precision_recall import precision_recall  # noqa: E402
from src.evaluation.summarize import POOLED, bootstrap_p, holm  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1", "macro_auc")
CLASS_METRICS = ("recall", "precision", "f1", "auc")
PATHS = {"full": ("main", SCENARIO_IDS), CROP_PATH: ("roi_crop", CROP_SCENARIO_IDS)}
PATH_LABELS = {"full": "Full image", CROP_PATH: "Per-crop ROI"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="More contrast minus original, paired bootstrap.")
    parser.add_argument("--original-root", type=Path, required=True, help="T23 runs (main/, roi_crop/).")
    parser.add_argument("--contrast-root", type=Path, required=True, help="T30 runs (main/, roi_crop/).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=list(FAMILY_ORDER),
                        help="Families to compare and pool (default: all 8).")
    return parser.parse_args(argv)


def scores(p: Predictions, idx: Optional[np.ndarray] = None) -> np.ndarray:
    """METRICS of a prediction file, or of the rows ``idx``."""
    t, y, pr = (p.y_true, p.y_pred, p.probs) if idx is None else (p.y_true[idx], p.y_pred[idx], p.probs[idx])
    s = summary_metrics(t, y, pr)
    return np.array([s["accuracy"], s["balanced_accuracy"], precision_recall(t, y)[0], s["macro_f1"], s["macro_auc"]])


def class_scores(p: Predictions, idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Per class (p.classes order) x CLASS_METRICS; 0 when undefined (as class_report), NaN for an absent class."""
    t, y, pr = (p.y_true, p.y_pred, p.probs) if idx is None else (p.y_true[idx], p.y_pred[idx], p.probs[idx])
    labels = list(range(len(p.classes)))
    precision, recall, f1, support = precision_recall_fscore_support(t, y, labels=labels, zero_division=0)
    out = np.column_stack([recall, precision, f1, [one_vs_rest_auc(t, pr, k) for k in labels]]).astype(float)
    out[support == 0] = np.nan
    return out


def check_pair(original: Predictions, contrast: Predictions, name: str) -> None:
    """Both files must score the same images of the same cases and classes."""
    key = ["relative_path", "pseudonimized_id", "true_class"]
    a = original.frame[key].sort_values(key).reset_index(drop=True)
    b = contrast.frame[key].sort_values(key).reset_index(drop=True)
    if not a.equals(b) or original.classes != contrast.classes:
        raise ValueError(f"{name}: the original and more-contrast test sets differ")


def paired(pairs: Dict[str, Tuple[Predictions, Predictions]], n_resamples: int, seed: int) -> Dict[str, Dict]:
    """Observed (original, contrast) scores per family, and contrast - original on each resample.

    Keys: ``observed`` / ``draws`` (METRICS) and ``class_observed`` / ``class_draws`` (classes x CLASS_METRICS).
    """
    families = list(pairs)
    case_ids = np.array(sorted(set(pairs[families[0]][0].cases)))
    rows_of = {(f, k): [np.flatnonzero(pairs[f][k].cases == c) for c in case_ids] for f in families for k in (0, 1)}
    n_classes = len(pairs[families[0]][0].classes)
    result = {"observed": {f: np.array([scores(pairs[f][0]), scores(pairs[f][1])]) for f in families},
              "class_observed": {f: np.array([class_scores(pairs[f][0]), class_scores(pairs[f][1])]) for f in families},
              "draws": {f: np.empty((n_resamples, len(METRICS))) for f in families},
              "class_draws": {f: np.empty((n_resamples, n_classes, len(CLASS_METRICS))) for f in families}}
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_resamples):
            picked = rng.integers(0, len(case_ids), size=len(case_ids))
            for f in families:
                idx = [np.concatenate([rows_of[(f, k)][i] for i in picked]) for k in (0, 1)]
                result["draws"][f][b] = scores(pairs[f][1], idx[1]) - scores(pairs[f][0], idx[0])
                result["class_draws"][f][b] = class_scores(pairs[f][1], idx[1]) - class_scores(pairs[f][0], idx[0])
    return result


def interval(d: np.ndarray) -> Tuple[float, float]:
    d = d[~np.isnan(d)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def comparison_rows(input_path: str, ids: Dict, observed: Dict, draws: Dict, n_cases: int, pooled_name: str = POOLED
                    ) -> List[Dict]:
    families = list(observed)
    rows = []
    for j, m in enumerate(METRICS):
        family_rows = []
        for f in families:
            original, contrast = observed[f][:, j]
            d = draws[f][:, j]
            low, high = interval(d)
            family_rows.append({"scenario_id": ids[(input_path, f)], "input_path": input_path, "family": f,
                                "metric": m, "original": original, "more_contrast": contrast,
                                "difference": contrast - original, "ci_low": low, "ci_high": high,
                                "p_value": bootstrap_p(d[~np.isnan(d)]), "test_cases": n_cases})
        for row, adjusted in zip(family_rows, holm([r["p_value"] for r in family_rows])):
            row["p_holm"] = adjusted
        pooled = np.nanmean(np.array([draws[f][:, j] for f in families]), axis=0)
        low, high = interval(pooled)
        original = float(np.mean([observed[f][0, j] for f in families]))
        contrast = float(np.mean([observed[f][1, j] for f in families]))
        rows += family_rows + [{"scenario_id": "", "input_path": input_path, "family": pooled_name, "metric": m,
                                "original": original, "more_contrast": contrast, "difference": contrast - original,
                                "ci_low": low, "ci_high": high, "p_value": bootstrap_p(pooled[~np.isnan(pooled)]),
                                "p_holm": np.nan, "test_cases": n_cases}]
    return rows


def class_rows(input_path: str, ids: Dict, pairs: Dict[str, Tuple[Predictions, Predictions]], observed: Dict,
               draws: Dict, pooled_name: str = POOLED) -> List[Dict]:
    """Per class, metric and family (and pooled): original, CLAHE, difference, interval, p (pooled: Holm over classes)."""
    families = list(pairs)
    first = pairs[families[0]][0]
    rows = []
    for m_index, m in enumerate(CLASS_METRICS):
        pooled_rows = []
        for k, name in enumerate(first.classes):
            mask = first.y_true == k
            common = {"input_path": input_path, "true_class": name, "metric": m,
                      "test_images": int(mask.sum()), "test_cases": len(set(first.cases[mask]))}
            for f in families:
                original, contrast = observed[f][:, k, m_index]
                d = draws[f][:, k, m_index]
                low, high = interval(d)
                rows.append({"scenario_id": ids[(input_path, f)], "family": f, **common, "original": original,
                             "more_contrast": contrast, "difference": contrast - original, "ci_low": low,
                             "ci_high": high, "p_value": bootstrap_p(d[~np.isnan(d)]), "p_holm": np.nan})
            pooled = np.nanmean(np.array([draws[f][:, k, m_index] for f in families]), axis=0)
            low, high = interval(pooled)
            original = float(np.nanmean([observed[f][0, k, m_index] for f in families]))
            contrast = float(np.nanmean([observed[f][1, k, m_index] for f in families]))
            pooled_rows.append({"scenario_id": "", "family": pooled_name, **common, "original": original,
                                "more_contrast": contrast, "difference": contrast - original, "ci_low": low,
                                "ci_high": high, "p_value": bootstrap_p(pooled[~np.isnan(pooled)])})
        for row, adjusted in zip(pooled_rows, holm([r["p_value"] for r in pooled_rows])):
            row["p_holm"] = adjusted
        rows += pooled_rows
    return rows


def change_rows(input_path: str, ids: Dict, pairs: Dict[str, Tuple[Predictions, Predictions]],
                pooled_name: str = POOLED) -> List[Dict]:
    """Per family and true class: images the CLAHE model gained (original wrong) and lost (original right)."""
    rows = []
    for f in pairs:
        original, contrast = (p.frame.assign(correct=p.y_true == p.y_pred).set_index("relative_path")
                              for p in pairs[f])
        both = original[["true_class", "correct"]].join(contrast["correct"], rsuffix="_contrast")
        for name, part in both.groupby("true_class", sort=False):
            rows.append({"scenario_id": ids[(input_path, f)], "input_path": input_path, "family": f,
                         "true_class": name, "test_images": len(part),
                         "original_correct": int(part["correct"].sum()),
                         "more_contrast_correct": int(part["correct_contrast"].sum()),
                         "gained": int((~part["correct"] & part["correct_contrast"]).sum()),
                         "lost": int((part["correct"] & ~part["correct_contrast"]).sum())})
    frame = pd.DataFrame(rows)
    pooled = frame.groupby("true_class", sort=False)[["test_images", "original_correct", "more_contrast_correct",
                                                      "gained", "lost"]].mean().reset_index()
    pooled = pooled.assign(scenario_id="", input_path=input_path, family=pooled_name)
    return rows + pooled[frame.columns].to_dict("records")


def plot_class_forest(table: pd.DataFrame, metric: str, path: Path, pooled_name: str = POOLED, n: int = 8) -> None:
    """Pooled per-class CLAHE minus original, one panel per input path."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    for ax, colour, (input_path, label) in zip(axes, CLASS_COLOURS, PATH_LABELS.items()):
        part = table[(table["input_path"] == input_path) & (table["metric"] == metric)
                     & (table["family"] == pooled_name)].reset_index(drop=True)
        y = np.arange(len(part))[::-1]
        ax.axvline(0, color=MUTED, lw=1)
        ax.errorbar(part["difference"], y, xerr=[part["difference"] - part["ci_low"], part["ci_high"] - part["difference"]],
                    fmt="none", ecolor=colour, elinewidth=2, capsize=0)
        ax.scatter(part["difference"], y, s=40, color=colour, zorder=3, edgecolors="white", linewidths=1.5)
        unit = "photos" if input_path == "full" else "crops"
        ax.set_yticks(y, [f"{c} ({n} {unit}, {p} pts)" for c, n, p in
                          zip(part["true_class"], part["test_images"], part["test_cases"])], fontsize=8)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color=MUTED, alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(f"{metric}: CLAHE - original, mean of {n}", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_heatmap(table: pd.DataFrame, metric: str, families: List[str], pooled_name: str, path: Path) -> None:
    """CLAHE minus original per wound class (rows) and family (columns, then their mean), one panel per input path.

    Blue = gain, orange = loss, grey = none; bold with * when the 95% interval excludes 0.
    """
    cmap = LinearSegmentedColormap.from_list("gain_loss", [CLASS_COLOURS[1], "#f0f0ee", CLASS_COLOURS[0]])
    columns = [*families, pooled_name]
    fig, axes = plt.subplots(1, 2, figsize=(1.15 * len(columns) * 2 + 3.5, 4.4))
    for ax, (input_path, label) in zip(axes, PATH_LABELS.items()):
        part = table[(table["input_path"] == input_path) & (table["metric"] == metric)]
        grid = part.pivot(index="true_class", columns="family", values="difference")
        classes = list(dict.fromkeys(part["true_class"]))
        grid = grid.loc[classes, columns]
        ax.imshow(grid.to_numpy(float), cmap=cmap, vmin=-0.15, vmax=0.15, aspect="auto")
        for i, c in enumerate(classes):
            for j, f in enumerate(columns):
                row = part[(part["true_class"] == c) & (part["family"] == f)].iloc[0]
                clear = row["ci_low"] > 0 or row["ci_high"] < 0
                ax.text(j, i, f"{row['difference']:+.2f}" + ("*" if clear else ""), ha="center", va="center",
                        fontsize=8, color=INK, fontweight="bold" if clear else "normal")
        unit = "photos" if input_path == "full" else "crops"
        sizes = part.drop_duplicates("true_class").set_index("true_class")
        ax.set_yticks(range(len(classes)), [f"{c} ({sizes.at[c, 'test_images']} {unit}, {sizes.at[c, 'test_cases']} pts)"
                                            for c in classes], fontsize=8)
        ax.set_xticks(range(len(columns)), [FAMILY_LABELS.get(f, f"Mean of {len(families)}") for f in columns],
                      rotation=35, ha="right", fontsize=8)
        ax.axvline(len(families) - 0.5, color="white", lw=3)
        ax.set_title(f"{label}: {metric}, CLAHE - original", fontsize=10, color=INK, loc="left")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_forest(table: pd.DataFrame, metric: str, path: Path, pooled_name: str = POOLED) -> None:
    """More contrast minus original per family and pooled, one panel per input path."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True)
    for ax, colour, (input_path, label) in zip(axes, CLASS_COLOURS, PATH_LABELS.items()):
        part = table[(table["input_path"] == input_path) & (table["metric"] == metric)].reset_index(drop=True)
        y = np.arange(len(part))[::-1]
        pooled = part["family"] == pooled_name
        ax.axvline(0, color=MUTED, lw=1)
        ax.errorbar(part["difference"], y, xerr=[part["difference"] - part["ci_low"], part["ci_high"] - part["difference"]],
                    fmt="none", ecolor=colour, elinewidth=2, capsize=0)
        ax.scatter(part["difference"], y, s=np.where(pooled, 70, 30), marker="o", color=colour, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.set_yticks(y, [(f"Mean of {len(part) - 1}" if f == pooled_name else f) for f in part["family"]], fontsize=8)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color=MUTED, alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(f"{metric.replace('_', ' ')}: more contrast - original", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; results stay outside it")
    out.mkdir(parents=True, exist_ok=True)

    # Families in the study's order; the pooled row is "all families (mean)" only when all 8 are compared
    families = [f for f in FAMILY_ORDER if f in args.families]
    pooled_name = POOLED if len(families) == len(FAMILY_ORDER) else f"mean of {len(families)} families"
    rows, per_class, changes = [], [], []
    for input_path, (run, ids) in PATHS.items():
        pairs = {}
        for f in families:
            original, contrast = (load_predictions(root / run / input_path / f / "test_predictions.csv", "test")
                                  for root in (args.original_root, args.contrast_root))
            check_pair(original, contrast, f"{input_path}/{f}")
            pairs[f] = (original, contrast)
        result = paired(pairs, args.bootstrap, args.seed)
        rows += comparison_rows(input_path, ids, result["observed"], result["draws"],
                                len(set(pairs[families[0]][0].cases)), pooled_name)
        per_class += class_rows(input_path, ids, pairs, result["class_observed"], result["class_draws"], pooled_name)
        changes += change_rows(input_path, ids, pairs, pooled_name)
        print(f"{input_path}: {len(families)} families paired", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / "contrast_vs_original.csv", index=False)
    for metric in ("accuracy", "macro_f1"):
        plot_forest(table, metric, out / f"contrast_vs_original_{metric}.png", pooled_name)
    class_table = pd.DataFrame(per_class)
    class_table.to_csv(out / "contrast_per_class.csv", index=False)
    pd.DataFrame(changes).to_csv(out / "contrast_per_class_changes.csv", index=False)
    for metric in ("recall", "f1"):
        plot_class_forest(class_table, metric, out / f"contrast_per_class_{metric}.png", pooled_name, len(families))
        plot_class_heatmap(class_table, metric, families, pooled_name, out / f"contrast_per_class_{metric}_by_model.png")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
