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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import CROP_PATH, CROP_SCENARIO_IDS, FAMILY_ORDER, SCENARIO_IDS  # noqa: E402
from src.evaluation.metrics import Predictions, load_predictions, one_vs_rest_auc, summary_metrics  # noqa: E402
from src.evaluation.plots import CLASS_COLOURS, INK, MUTED  # noqa: E402
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


def comparison_rows(input_path: str, ids: Dict, observed: Dict, draws: Dict, n_cases: int) -> List[Dict]:
    rows = []
    for j, m in enumerate(METRICS):
        family_rows = []
        for f in FAMILY_ORDER:
            original, contrast = observed[f][:, j]
            d = draws[f][:, j]
            low, high = interval(d)
            family_rows.append({"scenario_id": ids[(input_path, f)], "input_path": input_path, "family": f,
                                "metric": m, "original": original, "more_contrast": contrast,
                                "difference": contrast - original, "ci_low": low, "ci_high": high,
                                "p_value": bootstrap_p(d[~np.isnan(d)]), "test_cases": n_cases})
        for row, adjusted in zip(family_rows, holm([r["p_value"] for r in family_rows])):
            row["p_holm"] = adjusted
        pooled = np.nanmean(np.array([draws[f][:, j] for f in FAMILY_ORDER]), axis=0)
        low, high = interval(pooled)
        original = float(np.mean([observed[f][0, j] for f in FAMILY_ORDER]))
        contrast = float(np.mean([observed[f][1, j] for f in FAMILY_ORDER]))
        rows += family_rows + [{"scenario_id": "", "input_path": input_path, "family": POOLED, "metric": m,
                                "original": original, "more_contrast": contrast, "difference": contrast - original,
                                "ci_low": low, "ci_high": high, "p_value": bootstrap_p(pooled[~np.isnan(pooled)]),
                                "p_holm": np.nan, "test_cases": n_cases}]
    return rows


def class_rows(input_path: str, ids: Dict, pairs: Dict[str, Tuple[Predictions, Predictions]], observed: Dict,
               draws: Dict) -> List[Dict]:
    """Per class, metric and family (and pooled): original, CLAHE, difference, interval, p (pooled: Holm over classes)."""
    first = pairs[FAMILY_ORDER[0]][0]
    rows = []
    for m_index, m in enumerate(CLASS_METRICS):
        pooled_rows = []
        for k, name in enumerate(first.classes):
            mask = first.y_true == k
            common = {"input_path": input_path, "true_class": name, "metric": m,
                      "test_images": int(mask.sum()), "test_cases": len(set(first.cases[mask]))}
            for f in FAMILY_ORDER:
                original, contrast = observed[f][:, k, m_index]
                d = draws[f][:, k, m_index]
                low, high = interval(d)
                rows.append({"scenario_id": ids[(input_path, f)], "family": f, **common, "original": original,
                             "more_contrast": contrast, "difference": contrast - original, "ci_low": low,
                             "ci_high": high, "p_value": bootstrap_p(d[~np.isnan(d)]), "p_holm": np.nan})
            pooled = np.nanmean(np.array([draws[f][:, k, m_index] for f in FAMILY_ORDER]), axis=0)
            low, high = interval(pooled)
            original = float(np.nanmean([observed[f][0, k, m_index] for f in FAMILY_ORDER]))
            contrast = float(np.nanmean([observed[f][1, k, m_index] for f in FAMILY_ORDER]))
            pooled_rows.append({"scenario_id": "", "family": POOLED, **common, "original": original,
                                "more_contrast": contrast, "difference": contrast - original, "ci_low": low,
                                "ci_high": high, "p_value": bootstrap_p(pooled[~np.isnan(pooled)])})
        for row, adjusted in zip(pooled_rows, holm([r["p_value"] for r in pooled_rows])):
            row["p_holm"] = adjusted
        rows += pooled_rows
    return rows


def change_rows(input_path: str, ids: Dict, pairs: Dict[str, Tuple[Predictions, Predictions]]) -> List[Dict]:
    """Per family and true class: images the CLAHE model gained (original wrong) and lost (original right)."""
    rows = []
    for f in FAMILY_ORDER:
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
    pooled = pooled.assign(scenario_id="", input_path=input_path, family=POOLED)
    return rows + pooled[frame.columns].to_dict("records")


def plot_class_forest(table: pd.DataFrame, metric: str, path: Path) -> None:
    """Pooled per-class CLAHE minus original, one panel per input path."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    for ax, colour, (input_path, label) in zip(axes, CLASS_COLOURS, PATH_LABELS.items()):
        part = table[(table["input_path"] == input_path) & (table["metric"] == metric)
                     & (table["family"] == POOLED)].reset_index(drop=True)
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
        ax.set_xlabel(f"{metric}: CLAHE - original, mean of 8", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_forest(table: pd.DataFrame, metric: str, path: Path) -> None:
    """More contrast minus original per family and pooled, one panel per input path."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True)
    for ax, colour, (input_path, label) in zip(axes, CLASS_COLOURS, PATH_LABELS.items()):
        part = table[(table["input_path"] == input_path) & (table["metric"] == metric)].reset_index(drop=True)
        y = np.arange(len(part))[::-1]
        pooled = part["family"] == POOLED
        ax.axvline(0, color=MUTED, lw=1)
        ax.errorbar(part["difference"], y, xerr=[part["difference"] - part["ci_low"], part["ci_high"] - part["difference"]],
                    fmt="none", ecolor=colour, elinewidth=2, capsize=0)
        ax.scatter(part["difference"], y, s=np.where(pooled, 70, 30), marker="o", color=colour, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.set_yticks(y, [("Mean of 8" if f == POOLED else f) for f in part["family"]], fontsize=8)
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

    rows, per_class, changes = [], [], []
    for input_path, (run, ids) in PATHS.items():
        pairs = {}
        for f in FAMILY_ORDER:
            original, contrast = (load_predictions(root / run / input_path / f / "test_predictions.csv", "test")
                                  for root in (args.original_root, args.contrast_root))
            check_pair(original, contrast, f"{input_path}/{f}")
            pairs[f] = (original, contrast)
        result = paired(pairs, args.bootstrap, args.seed)
        rows += comparison_rows(input_path, ids, result["observed"], result["draws"],
                                len(set(pairs[FAMILY_ORDER[0]][0].cases)))
        per_class += class_rows(input_path, ids, pairs, result["class_observed"], result["class_draws"])
        changes += change_rows(input_path, ids, pairs)
        print(f"{input_path}: {len(FAMILY_ORDER)} families paired", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / "contrast_vs_original.csv", index=False)
    for metric in ("accuracy", "macro_f1"):
        plot_forest(table, metric, out / f"contrast_vs_original_{metric}.png")
    class_table = pd.DataFrame(per_class)
    class_table.to_csv(out / "contrast_per_class.csv", index=False)
    pd.DataFrame(changes).to_csv(out / "contrast_per_class_changes.csv", index=False)
    for metric in ("recall", "f1"):
        plot_class_forest(class_table, metric, out / f"contrast_per_class_{metric}.png")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
