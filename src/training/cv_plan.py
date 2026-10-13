#!/usr/bin/env python3
"""List every T12b cross-validation job, with each scenario's settings read from the main run.

S01-S12 use the T08b chosen learning rate and train for the selected run's best epoch; S13-S14
fit the T06 chosen SVM; S15-S16 train members with seeds 43-45 for each member's best epoch,
then soft-vote them per fold. Standard library only, so it runs on the login node.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


INPUT_PATHS = ("full", "roi")
FAMILY_ORDER = ("efficientnet_b0", "resnet152", "mobilenet_v3_small", "googlenet",
                "vgg16", "vit_b16", "cnn_svm", "alexnet_ensemble")  # same as evaluate.py
TUNED = FAMILY_ORDER[:6]
COLUMNS = ("job_name", "kind", "family", "input_path", "fold", "lr", "epochs", "seed",
           "svm_params_from", "output_dir")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan the T12b cross-validation jobs.")
    parser.add_argument("--main-root", type=Path, required=True, help="Finished main run, e.g. .../experiments/main")
    parser.add_argument("--cv-root", type=Path, required=True, help="CV output root, e.g. .../experiments/cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True, help="Plan TSV to write.")
    return parser.parse_args(argv)


def scenario_id(input_path: str, family: str) -> str:
    return f"S{2 * FAMILY_ORDER.index(family) + INPUT_PATHS.index(input_path) + 1:02d}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; is the main run finished?")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def selected_run(scenario: Path) -> Dict[str, str]:
    rows = [r for r in read_csv(scenario / "tuning_summary.csv") if r["selected"] == "True"]
    if len(rows) != 1:
        raise ValueError(f"{scenario}/tuning_summary.csv must have exactly one selected row")
    return rows[0]


def member_runs(ensemble: Path) -> List[Dict[str, object]]:
    members = []
    for row in read_csv(ensemble / "ensemble_members.csv"):
        config = json.loads((Path(row["output_dir"]) / "run_config.json").read_text(encoding="utf-8"))
        members.append({"member": int(row["member"]), "seed": int(row["seed"]),
                        "epochs": int(row["best_epoch"]), "lr": config["lr"]})
    return sorted(members, key=lambda m: m["member"])


def plan(main_root: Path, cv_root: Path, folds: int) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    for family in FAMILY_ORDER:
        for path in INPUT_PATHS:
            sid, scenario = scenario_id(path, family), main_root / path / family
            for k in range(1, folds + 1):
                out = cv_root / path / family / f"fold_{k}"
                name = f"CV_{sid}_{family}_{path}_f{k}"
                base = {"family": family, "input_path": path, "fold": k, "lr": "", "epochs": "", "seed": 42,
                        "svm_params_from": "", "output_dir": str(out)}
                if family in TUNED:
                    chosen = selected_run(scenario)
                    jobs.append({**base, "job_name": name, "kind": "train",
                                 "lr": chosen["lr"], "epochs": int(chosen["best_epoch"])})
                elif family == "cnn_svm":
                    config = scenario / "run_config.json"
                    if not config.is_file():
                        raise FileNotFoundError(f"{config} not found; is the main run finished?")
                    jobs.append({**base, "job_name": name, "kind": "svm", "svm_params_from": str(config)})
                else:
                    for m in member_runs(scenario):
                        jobs.append({**base, "job_name": f"{name}_m{m['member']}", "kind": "member",
                                     "family": "alexnet", "lr": m["lr"], "epochs": m["epochs"], "seed": m["seed"],
                                     "output_dir": str(out / f"member_{m['member']:02d}")})
                    jobs.append({**base, "job_name": f"{name}_combine", "kind": "combine"})
    return jobs


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    jobs = plan(args.main_root, args.cv_root, args.folds)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(jobs)
    kinds = {k: sum(j["kind"] == k for j in jobs) for k in ("train", "svm", "member", "combine")}
    print(f"{len(jobs)} CV jobs ({kinds}) written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
