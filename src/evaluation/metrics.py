"""Metrics computed from per-image prediction files, never from a live model."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


ID_COLUMNS = ["relative_path", "pseudonimized_id", "split", "true_class", "predicted_class"]
PROB_PREFIX = "prob_"
BOOTSTRAP_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")


@dataclass
class Predictions:
    frame: pd.DataFrame
    classes: List[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray
    cases: np.ndarray

    def signature(self) -> Counter:
        """Images per (case, true class): equal signatures mean the same evaluation set."""
        return Counter(zip(self.frame["pseudonimized_id"], self.frame["true_class"]))


def load_predictions(path: Path, split: str) -> Predictions:
    """Read a prediction CSV and check it is internally consistent."""
    df = pd.read_csv(path, dtype={c: str for c in ID_COLUMNS})
    missing = [c for c in ID_COLUMNS if c not in df.columns]
    prob_columns = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    if missing or not prob_columns:
        raise ValueError(f"{path}: missing columns {missing or 'prob_*'}")
    if df["relative_path"].duplicated().any():
        dupes = df.loc[df["relative_path"].duplicated(), "relative_path"].tolist()
        raise ValueError(f"{path}: duplicate rows for {dupes[:5]}")
    if not (df["split"] == split).all():
        raise ValueError(f"{path}: expected split {split!r}, found {sorted(df['split'].unique())}")

    classes = [c[len(PROB_PREFIX):].replace("_", " ") for c in prob_columns]
    unknown = sorted(set(df["true_class"]) - set(classes))
    if unknown:
        raise ValueError(f"{path}: true classes {unknown} have no probability column")

    probs = df[prob_columns].to_numpy(dtype=float)
    if (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{path}: probabilities must be non-negative and sum to 1 per row")
    argmax = np.array(classes)[probs.argmax(axis=1)]
    if not (argmax == df["predicted_class"].to_numpy()).all():
        raise ValueError(f"{path}: predicted_class differs from the argmax of the probabilities")

    index = {name: i for i, name in enumerate(classes)}
    return Predictions(
        frame=df,
        classes=classes,
        y_true=df["true_class"].map(index).to_numpy(),
        y_pred=probs.argmax(axis=1),
        probs=probs / probs.sum(axis=1, keepdims=True),
        cases=df["pseudonimized_id"].to_numpy(),
    )


def one_vs_rest_auc(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    positives = y_true == k
    if positives.all() or not positives.any():
        return float("nan")
    return float(roc_auc_score(positives, probs[:, k]))


def roc_points(p: Predictions) -> pd.DataFrame:
    """One-vs-rest ROC points per class plus their macro average, in long format.

    Columns: ``curve`` (class name or ``macro average``), ``fpr``, ``tpr``, ``threshold``, ``auc``.
    A class with no positives or no negatives in the split has no curve. The macro curve is the
    mean of the class curves interpolated on the union of their false positive rates; its ``auc``
    is the mean of the class AUCs, the same number as ``macro_auc`` in the metrics JSON.
    """
    frames, class_curves, aucs = [], [], []
    for k, name in enumerate(p.classes):
        auc = one_vs_rest_auc(p.y_true, p.probs, k)
        if np.isnan(auc):
            continue
        fpr, tpr, threshold = roc_curve(p.y_true == k, p.probs[:, k])
        frames.append(pd.DataFrame({"curve": name, "fpr": fpr, "tpr": tpr, "threshold": threshold, "auc": auc}))
        class_curves.append((fpr, tpr))
        aucs.append(auc)
    if not class_curves:
        raise ValueError("No class has both positives and negatives; ROC curves are undefined")
    grid = np.unique(np.concatenate([fpr for fpr, _ in class_curves]))
    mean_tpr = np.mean([np.interp(grid, fpr, tpr) for fpr, tpr in class_curves], axis=0)
    frames.append(pd.DataFrame({"curve": "macro average", "fpr": grid, "tpr": mean_tpr,
                                "threshold": np.nan, "auc": float(np.mean(aucs))}))
    return pd.concat(frames, ignore_index=True)


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Accuracy, balanced accuracy, macro F1 and macro one-vs-rest AUC.

    Averages run over the classes present in ``y_true``. With every class present (the real
    test set) this is the usual definition; in a bootstrap resample that misses a rare class
    the missing class is left out rather than counted as zero.
    """
    present = np.unique(y_true)
    _, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_auc": float(np.nanmean([one_vs_rest_auc(y_true, probs, k) for k in present])),
    }


def compute_metrics(p: Predictions) -> Dict[str, float]:
    labels = list(range(len(p.classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        p.y_true, p.y_pred, labels=labels, zero_division=0
    )
    aucs = np.array([one_vs_rest_auc(p.y_true, p.probs, k) for k in labels])
    weights = support / support.sum()
    present = support > 0
    return {
        **summary_metrics(p.y_true, p.y_pred, p.probs),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "weighted_auc": float(np.nansum(aucs * weights)),
        "log_loss": float(log_loss(p.y_true, p.probs, labels=labels)),
        "n_images": int(len(p.y_true)),
        "n_cases": int(len(set(p.cases))),
    }


def class_report(p: Predictions) -> pd.DataFrame:
    """Class-wise precision, recall, F1 and AUC with image and case support, plus averages."""
    labels = list(range(len(p.classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        p.y_true, p.y_pred, labels=labels, zero_division=0
    )
    rows = []
    for k, name in enumerate(p.classes):
        mask = p.y_true == k
        rows.append({
            "class": name,
            "precision": precision[k],
            "recall": recall[k],
            "f1": f1[k],
            "auc": one_vs_rest_auc(p.y_true, p.probs, k),
            "support_images": int(support[k]),
            "support_cases": int(len(set(p.cases[mask]))),
        })
    table = pd.DataFrame(rows)
    totals = {"support_images": int(support.sum()), "support_cases": int(len(set(p.cases)))}
    weights = table["support_images"] / table["support_images"].sum()
    macro = {"class": "macro avg", **table[["precision", "recall", "f1", "auc"]].mean().to_dict(), **totals}
    weighted = {"class": "weighted avg",
                **(table[["precision", "recall", "f1", "auc"]].mul(weights, axis=0).sum()).to_dict(), **totals}
    return pd.concat([table, pd.DataFrame([macro, weighted])], ignore_index=True)


def confusion_table(p: Predictions) -> pd.DataFrame:
    matrix = confusion_matrix(p.y_true, p.y_pred, labels=list(range(len(p.classes))))
    return pd.DataFrame(matrix, index=pd.Index(p.classes, name="true_class"), columns=p.classes)


def bootstrap_draws(p: Predictions, n_resamples: int = 2000, seed: int = 42) -> Dict[str, np.ndarray]:
    """Summary metrics on resamples of cases drawn with replacement; a case's images stay together."""
    rng = np.random.default_rng(seed)
    case_ids = np.array(sorted(set(p.cases)))
    rows_of = [np.flatnonzero(p.cases == c) for c in case_ids]
    draws: Dict[str, List[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_resamples):
            picked = rng.integers(0, len(case_ids), size=len(case_ids))
            idx = np.concatenate([rows_of[i] for i in picked])
            values = summary_metrics(p.y_true[idx], p.y_pred[idx], p.probs[idx])
            for name in BOOTSTRAP_METRICS:
                draws[name].append(values[name])
    return {name: np.array(values) for name, values in draws.items()}


def case_bootstrap(p: Predictions, n_resamples: int = 2000, seed: int = 42,
                   level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """Percentile confidence intervals from the case-level bootstrap."""
    tail = (1.0 - level) / 2 * 100
    result = {}
    for name, values in bootstrap_draws(p, n_resamples, seed).items():
        values = values[~np.isnan(values)]
        result[name] = {
            "low": float(np.percentile(values, tail)),
            "high": float(np.percentile(values, 100 - tail)),
            "n_resamples": int(len(values)),
        }
    return result
