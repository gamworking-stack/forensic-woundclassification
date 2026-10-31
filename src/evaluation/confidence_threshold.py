#!/usr/bin/env python3
"""T29: test results with an "inconclusive" answer below a confidence threshold (Halloween experiment).

A prediction counts only when its top softmax probability reaches ``--threshold`` (0.60); below it the
image (full image, S01-S15) or crop (per-crop, S25-S32) is inconclusive. Per scenario this reports how
many images become inconclusive and how many of those the model had right or wrong, then the Table 2
metrics (accuracy, balanced accuracy, macro precision, macro recall, macro-F1, macro AUC) on the
conclusive images, and accuracy over every image with inconclusive counted as not correct.

Intervals use the T08 resamples (``bootstrap_draws``: sorted cases, ``default_rng(seed)``); each
resample keeps its conclusive rows and is scored on them. All families of an input path share the test
set and so the resamples; the pooled row is the mean over the 8 families, its interval from the
per-resample means (T15). With threshold 0 every image is conclusive, so the script first checks that
it reproduces each saved ``test_metrics.json`` and its intervals. Run it on the T25 copies
(``patient_rescore.py``) for patient-level intervals. Nothing is retrained or re-predicted; the
threshold was set by the researcher, not tuned on the test set. Outputs hold no per-image rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import CROP_PATH, CROP_SCENARIO_IDS, FAMILY_ORDER, SCENARIO_IDS  # noqa: E402
from src.evaluation.metrics import Predictions, load_predictions, summary_metrics  # noqa: E402
from src.evaluation.plots import CLASS_COLOURS, INK, MUTED  # noqa: E402
from src.evaluation.precision_recall import precision_recall  # noqa: E402
from src.evaluation.summarize import POOLED  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
INCONCLUSIVE = "inconclusive"
METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc")
# coverage: share conclusive; accuracy_all: correct and conclusive over every image; then METRICS on the conclusive
VALUES = ("coverage", "accuracy_all", *METRICS)
SAVED_CI = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")  # intervals in test_metrics.json
PATH_LABELS = {"full": "Full image (S01-S15)", CROP_PATH: "Per-crop ROI (S25-S32)"}
CURVE_THRESHOLDS = np.round(np.arange(0.0, 0.96, 0.05), 2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test metrics with inconclusive predictions below a threshold.")
    parser.add_argument("--main-root", type=Path, required=True, help="Evaluated main run (full/<family>/).")
    parser.add_argument("--crop-root", type=Path, required=True, help="Evaluated per-crop run (crop/<family>/).")
    parser.add_argument("--reference-main", type=Path, required=True,
                        help="Folder with full/<family>/test_metrics.json to reproduce at threshold 0.")
    parser.add_argument("--reference-crop", type=Path, required=True,
                        help="Folder with crop/<family>/test_metrics.json to reproduce at threshold 0.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--sweep", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def values(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, conclusive: np.ndarray) -> np.ndarray:
    """VALUES for one sample: coverage, accuracy over all images, METRICS on the conclusive ones (NaN if none)."""
    head = [conclusive.mean(), ((y_true == y_pred) & conclusive).mean()]
    if not conclusive.any():
        return np.array(head + [np.nan] * len(METRICS))
    t, p, pr = y_true[conclusive], y_pred[conclusive], probs[conclusive]
    s = summary_metrics(t, p, pr)
    precision, recall = precision_recall(t, p)
    return np.array(head + [s["accuracy"], s["balanced_accuracy"], precision, recall, s["macro_f1"], s["macro_auc"]])


def draws(p: Predictions, conclusive: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """VALUES on ``bootstrap_draws``' case resamples: shape (n_resamples, len(VALUES))."""
    rng = np.random.default_rng(seed)
    case_ids = np.array(sorted(set(p.cases)))
    rows_of = [np.flatnonzero(p.cases == c) for c in case_ids]
    out = np.empty((n_resamples, len(VALUES)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_resamples):
            picked = rng.integers(0, len(case_ids), size=len(case_ids))
            idx = np.concatenate([rows_of[i] for i in picked])
            out[b] = values(p.y_true[idx], p.y_pred[idx], p.probs[idx], conclusive[idx])
    return out


def interval(column: np.ndarray) -> Tuple[float, float]:
    column = column[~np.isnan(column)]
    return float(np.percentile(column, 2.5)), float(np.percentile(column, 97.5))


def check_reference(p: Predictions, saved_path: Path, n_resamples: int, seed: int) -> None:
    """Threshold 0 (every image conclusive) must reproduce the saved metrics and intervals."""
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    everything = np.ones(len(p.y_true), dtype=bool)
    observed = dict(zip(VALUES, values(p.y_true, p.y_pred, p.probs, everything)))
    sampled = draws(p, everything, n_resamples, seed)
    for name in (*SAVED_CI, "macro_precision", "macro_recall"):
        if not np.isclose(observed[name], saved[name]):
            raise ValueError(f"{saved_path}: {name} {observed[name]} differs from the saved {saved[name]}")
    for name in SAVED_CI:
        ci = saved["confidence_intervals"][name]
        if not np.allclose(interval(sampled[:, VALUES.index(name)]), (ci["low"], ci["high"])):
            raise ValueError(f"{saved_path}: the {name} interval differs from the saved one")


def scenario_row(common: Dict, p: Predictions, conclusive: np.ndarray, observed: np.ndarray,
                 sampled: np.ndarray) -> Dict:
    correct = p.y_true == p.y_pred
    row = {**common, "test_rows": len(p.y_true), "test_cases": len(set(p.cases)),
           "inconclusive": int((~conclusive).sum()),
           "inconclusive_would_be_correct": int((~conclusive & correct).sum()),
           "inconclusive_would_be_wrong": int((~conclusive & ~correct).sum()),
           "conclusive": int(conclusive.sum()), "conclusive_correct": int((conclusive & correct).sum()),
           "accuracy_no_threshold": float(correct.mean())}
    for j, name in enumerate(VALUES):
        low, high = interval(sampled[:, j])
        row.update({name: float(observed[j]), f"{name}_ci_low": low, f"{name}_ci_high": high})
    return row


def class_rows(common: Dict, p: Predictions, conclusive: np.ndarray) -> List[Dict]:
    """Per true class: images, inconclusive, and accuracy (= recall) on the conclusive and on all images."""
    rows = []
    correct = p.y_true == p.y_pred
    for k, name in enumerate(p.classes):
        mask = p.y_true == k
        kept = mask & conclusive
        rows.append({**common, "true_class": name, "test_rows": int(mask.sum()),
                     "inconclusive": int((mask & ~conclusive).sum()), "conclusive": int(kept.sum()),
                     "conclusive_correct": int((kept & correct).sum()),
                     "accuracy_conclusive": float((kept & correct).sum() / kept.sum()) if kept.any() else np.nan,
                     "accuracy_all": float((kept & correct).sum() / mask.sum()) if mask.any() else np.nan,
                     "accuracy_no_threshold": float((mask & correct).sum() / mask.sum()) if mask.any() else np.nan})
    return rows


def confusion_rows(common: Dict, p: Predictions, conclusive: np.ndarray) -> pd.DataFrame:
    """Confusion counts with an extra ``inconclusive`` column."""
    predicted = np.where(conclusive, np.array(p.classes)[p.y_pred], INCONCLUSIVE)
    table = pd.crosstab(pd.Categorical(np.array(p.classes)[p.y_true], categories=p.classes),
                        pd.Categorical(predicted, categories=[*p.classes, INCONCLUSIVE]), dropna=False)
    table.index.name, table.columns.name = "true_class", None
    out = table.reset_index()
    for key, value in reversed(list(common.items())):
        out.insert(0, key, value)
    return out


def pooled_class_rows(rows: List[Dict], common: Dict) -> List[Dict]:
    """Mean over families per true class (counts become means too)."""
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c not in ("scenario_id", "input_path", "family", "threshold", "true_class")]
    means = frame.groupby("true_class", sort=False)[numeric].mean().reset_index()
    return [{**common, **r} for r in means.to_dict("records")]


def plot_curve(curve: pd.DataFrame, threshold: float, path: Path) -> None:
    """Pooled accuracy on conclusive images against coverage, one line per input path."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for colour, (input_path, label) in zip(CLASS_COLOURS, PATH_LABELS.items()):
        part = curve[curve["input_path"] == input_path].sort_values("threshold")
        at = part[np.isclose(part["threshold"], threshold)].iloc[0]
        # The chosen threshold's values go in the legend, so no label crowds the curves
        ax.plot(part["coverage"], part["accuracy"], color=colour, lw=2, marker="o", ms=4,
                label=f"{label}\nat {threshold:.2f} (large dot): accuracy {at['accuracy']:.3f}, coverage {at['coverage']:.0%}")
        ax.plot(at["coverage"], at["accuracy"], marker="o", ms=9, color=colour, mec="white", mew=2, zorder=3)
    ax.set_xlabel("Coverage (share of test images given a class)", color=INK)
    ax.set_ylabel("Accuracy on conclusive images", color=INK)
    ax.set_title("Confidence threshold 0 to 0.95 (mean of 8 families)", fontsize=10, color=INK, loc="left")
    ax.invert_xaxis()
    ax.grid(color=MUTED, alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelspacing=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; results stay outside it")
    out.mkdir(parents=True, exist_ok=True)

    summary, per_class, confusion, sweep = [], [], [], []
    paths = (("full", args.main_root, args.reference_main, SCENARIO_IDS),
             (CROP_PATH, args.crop_root, args.reference_crop, CROP_SCENARIO_IDS))
    for input_path, root, reference, ids in paths:
        predictions = {f: load_predictions(root / input_path / f / "test_predictions.csv", "test") for f in FAMILY_ORDER}
        first = predictions[FAMILY_ORDER[0]]
        for f, p in predictions.items():
            if p.signature() != first.signature():
                raise ValueError(f"{input_path}/{f} was scored on another test set than {FAMILY_ORDER[0]}")
            check_reference(p, reference / input_path / f / "test_metrics.json", args.bootstrap, args.seed)
        print(f"{input_path}: threshold 0 reproduces every saved test_metrics.json and interval", flush=True)

        # The chosen threshold, with intervals; the pooled row from the per-resample family means
        observed, sampled, classes = {}, {}, []
        for f, p in predictions.items():
            conclusive = p.probs.max(axis=1) >= args.threshold
            common = {"scenario_id": ids[(input_path, f)], "input_path": input_path, "family": f,
                      "threshold": args.threshold}
            observed[f] = values(p.y_true, p.y_pred, p.probs, conclusive)
            sampled[f] = draws(p, conclusive, args.bootstrap, args.seed)
            summary.append(scenario_row(common, p, conclusive, observed[f], sampled[f]))
            classes += class_rows(common, p, conclusive)
            confusion.append(confusion_rows(common, p, conclusive))
            print(f"{common['scenario_id']} {f:<20} inconclusive {summary[-1]['inconclusive']:>3}/{len(p.y_true)}  "
                  f"accuracy on conclusive {observed[f][VALUES.index('accuracy')]:.3f}", flush=True)
        pooled = {"scenario_id": "", "input_path": input_path, "family": POOLED, "threshold": args.threshold}
        counts = pd.DataFrame(summary[-len(FAMILY_ORDER):]).drop(columns=list(pooled)).mean()
        row = {**pooled, **counts.to_dict()}
        pooled_draws = np.mean([sampled[f] for f in FAMILY_ORDER], axis=0)
        for j, name in enumerate(VALUES):
            low, high = interval(pooled_draws[:, j])
            row.update({name: float(np.mean([observed[f][j] for f in FAMILY_ORDER])),
                        f"{name}_ci_low": low, f"{name}_ci_high": high})
        summary.append(row)
        per_class += classes + pooled_class_rows(classes, pooled)

        # Point estimates at other thresholds (no intervals), per family and pooled
        for t in sorted({0.0, *args.sweep, *CURVE_THRESHOLDS}):
            scores = {f: values(p.y_true, p.y_pred, p.probs, p.probs.max(axis=1) >= t) for f, p in predictions.items()}
            for f in [*FAMILY_ORDER, POOLED]:
                v = np.mean(list(scores.values()), axis=0) if f == POOLED else scores[f]
                n = len(first.y_true)
                sweep.append({"scenario_id": ids.get((input_path, f), ""), "input_path": input_path, "family": f,
                              "threshold": float(t), "inconclusive": n - v[0] * n, **dict(zip(VALUES, v))})

    sweep_frame = pd.DataFrame(sweep)
    pd.DataFrame(summary).to_csv(out / "threshold_summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "threshold_per_class.csv", index=False)
    pd.concat(confusion, ignore_index=True).to_csv(out / "threshold_confusion.csv", index=False)
    in_sweep = sweep_frame["threshold"].round(2).isin(np.round([0.0, *args.sweep], 2))
    sweep_frame[in_sweep].to_csv(out / "threshold_sweep.csv", index=False)
    plot_curve(sweep_frame[sweep_frame["family"] == POOLED], args.threshold, out / "threshold_coverage.png")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
