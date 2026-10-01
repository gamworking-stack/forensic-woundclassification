#!/usr/bin/env python3
"""Evaluate prediction files: one scenario folder, or every scenario in a run.

Per scenario and split it writes metrics JSON, a class-wise report, and the confusion matrix
(CSV and PNG); test metrics get case-level bootstrap confidence intervals. For a whole run it
also checks that every scenario was tested on the same images and writes evaluation_index.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import (  # noqa: E402
    case_bootstrap,
    class_report,
    compute_metrics,
    confusion_table,
    load_predictions,
)
from src.evaluation.plots import plot_confusion_matrix  # noqa: E402


SPLITS = ("validation", "test")
INPUT_PATHS = ("full", "roi")
# Scenario IDs from spec §5.1: family order, full then ROI.
FAMILY_ORDER = (
    "efficientnet_b0", "resnet152", "mobilenet_v3_small", "googlenet",
    "vgg16", "vit_b16", "cnn_svm", "alexnet_ensemble",
)
SCENARIO_IDS = {
    (input_path, family): f"S{2 * i + j + 1:02d}"
    for i, family in enumerate(FAMILY_ORDER)
    for j, input_path in enumerate(INPUT_PATHS)
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction files for one or all scenarios.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", type=Path, help="One scenario folder with prediction files.")
    target.add_argument("--experiments-root", type=Path,
                        help="A run folder laid out as <full|roi>/<family>/.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Case-level resamples for test CIs.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def evaluate_scenario(run_dir: Path, title: str, n_resamples: int, seed: int) -> Dict[str, Any]:
    """Write all metric files for one scenario; return its test metrics."""
    results: Dict[str, Any] = {}
    for split in SPLITS:
        p = load_predictions(run_dir / f"{split}_predictions.csv", split)
        metrics: Dict[str, Any] = {"scenario": title, "split": split, **compute_metrics(p)}
        if split == "test":
            metrics["confidence_intervals"] = case_bootstrap(p, n_resamples, seed)
            metrics["bootstrap"] = {"unit": "case", "resamples": n_resamples, "seed": seed, "level": 0.95}
        with (run_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        class_report(p).to_csv(run_dir / f"{split}_classification_report.csv", index=False)
        table = confusion_table(p)
        table.to_csv(run_dir / f"{split}_confusion_matrix.csv")
        plot_confusion_matrix(table, f"{title} | {split}", run_dir / f"{split}_confusion_matrix.png")
        results[split] = metrics
    return results["test"]


def find_scenarios(root: Path) -> List[Tuple[str, str, Path]]:
    """(input_path, family, folder) for every <full|roi>/<family>/ holding prediction files.

    Only direct children count, so ensemble member folders one level deeper are skipped.
    """
    found = []
    for input_path in INPUT_PATHS:
        base = root / input_path
        if not base.is_dir():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            if all((folder / f"{split}_predictions.csv").is_file() for split in SPLITS):
                found.append((input_path, folder.name, folder))
    return found


def check_same_test_set(scenarios: List[Tuple[str, str, Path]]) -> None:
    """Every scenario must be tested on the same cases with the same images per class."""
    signatures = {
        f"{input_path}/{family}": load_predictions(folder / "test_predictions.csv", "test").signature()
        for input_path, family, folder in scenarios
    }
    names = list(signatures)
    reference = names[0]
    different = [n for n in names[1:] if signatures[n] != signatures[reference]]
    if different:
        raise ValueError(f"Test sets differ from {reference}: {different}. "
                         "Runs with another scope belong in their own experiments root.")


def evaluate_all(root: Path, n_resamples: int, seed: int) -> pd.DataFrame:
    scenarios = find_scenarios(root)
    if not scenarios:
        raise FileNotFoundError(f"No scenario folders with prediction files under {root}")
    check_same_test_set(scenarios)

    rows = []
    for input_path, family, folder in scenarios:
        scenario_id = SCENARIO_IDS.get((input_path, family), "")
        title = f"{scenario_id} {family} ({input_path})".strip()
        test = evaluate_scenario(folder, title, n_resamples, seed)
        ci = test["confidence_intervals"]["macro_f1"]
        rows.append({
            "scenario_id": scenario_id,
            "input_path": input_path,
            "family": family,
            "status": "evaluated",
            "test_images": test["n_images"],
            "test_cases": test["n_cases"],
            "test_macro_f1": test["macro_f1"],
            "test_macro_f1_ci_low": ci["low"],
            "test_macro_f1_ci_high": ci["high"],
            "test_balanced_accuracy": test["balanced_accuracy"],
            "test_macro_auc": test["macro_auc"],
            "folder": str(folder),
        })
        print(f"{title}: test macro-F1 {test['macro_f1']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]", flush=True)

    done = {(r["input_path"], r["family"]) for r in rows}
    for (input_path, family), scenario_id in SCENARIO_IDS.items():
        if (input_path, family) not in done:
            rows.append({"scenario_id": scenario_id, "input_path": input_path, "family": family,
                         "status": "missing"})
    index = pd.DataFrame(rows)
    index["sort_key"] = index["scenario_id"].replace("", "S99")
    index = index.sort_values(["sort_key", "input_path", "family"]).drop(columns="sort_key")
    index.to_csv(root / "evaluation_index.csv", index=False)
    return index


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.run_dir is not None:
        test = evaluate_scenario(args.run_dir, args.run_dir.name, args.bootstrap, args.seed)
        ci = test["confidence_intervals"]["macro_f1"]
        print(f"{args.run_dir}: test macro-F1 {test['macro_f1']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]")
        return 0
    index = evaluate_all(args.experiments_root, args.bootstrap, args.seed)
    counts = index["status"].value_counts().to_dict()
    print(f"Evaluated {counts.get('evaluated', 0)} scenarios, missing {counts.get('missing', 0)}; "
          f"index written to {args.experiments_root / 'evaluation_index.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
