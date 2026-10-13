#!/usr/bin/env python3
"""Pick a scenario's learning rate on validation, then predict test for that run only (T08b).

Each grid value is an ordinary ``train.py --skip-test`` run in ``<scenario>/lr_<value>/``. The run
with the highest validation macro-F1 at its best epoch wins (ties: lower validation loss, then the
smaller learning rate). The choice is written to ``tuning_summary.csv`` before the test split is
touched; then the chosen run is resumed without ``--skip-test``, which predicts validation and
test from its ``best.pt`` without training, and its files are copied into the scenario folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.common import Logger  # noqa: E402
from src.training.engine import load_checkpoint  # noqa: E402
from src.training.train import main as train_main  # noqa: E402


LR_GRID = ("1e-5", "3e-5", "1e-4")  # fixed 2026-10-13, before any S01-S12 run (T08b §2)
COPIED_FILES = ("run_config.json", "training_history.csv", "validation_predictions.csv",
                "test_predictions.csv")  # test last: it marks the scenario finished


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose a scenario's learning rate on validation.")
    parser.add_argument("--scenario-dir", type=Path, required=True,
                        help="Scenario folder holding the lr_<value>/ grid runs.")
    parser.add_argument("--lrs", nargs="+", default=list(LR_GRID))
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers for the test prediction (default: as trained).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def read_grid_run(folder: Path, lr: str) -> Dict[str, Any]:
    """Best-epoch validation scores of one finished grid run."""
    for name in ("best.pt", "validation_predictions.csv", "run_config.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"{folder} has no {name}; has its tuning run finished?")
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    if float(config["lr"]) != float(lr):
        raise ValueError(f"{folder} was trained with lr={config['lr']}, expected {lr}")
    best = load_checkpoint(folder / "best.pt", torch.device("cpu"))
    return {
        "lr": lr,
        "best_epoch": int(best["epoch"]),
        "validation_macro_f1": float(best["validation_macro_f1"]),
        "validation_loss": float(best["validation_loss"]),
        "output_dir": str(folder),
        "config": config,
    }


def choose(runs: List[Dict[str, Any]]) -> int:
    """Index of the winning run: highest macro-F1, then lower loss, then smaller learning rate."""
    return max(range(len(runs)), key=lambda i: (runs[i]["validation_macro_f1"],
                                                -runs[i]["validation_loss"],
                                                -float(runs[i]["lr"])))


def resume_args(config: Dict[str, Any], num_workers: Optional[int], device: str) -> List[str]:
    """train.py arguments that reproduce a finished run, resumed, with the test split included."""
    args = [
        "--family", config["family"],
        "--input-path", config["input_path"],
        "--manifest", config["manifest"],
        "--image-root", config["image_root"],
        "--output-dir", config["output_dir"],
        "--roi-manifest", config["roi_manifest"],
        "--full-scope", config["full_scope"],
        "--image-size", str(config["image_size"]),
        "--batch-size", str(config["batch_size"]),
        "--max-epochs", str(config["max_epochs"]),
        "--patience", str(config["patience"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--label-smoothing", str(config["label_smoothing"]),
        "--seed", str(config["seed"]),
        "--num-workers", str(config["num_workers"] if num_workers is None else num_workers),
        "--device", device,
        "--resume",
    ]
    for flag, enabled in (("--no-class-weights", not config["class_weights"]),
                          ("--no-pretrained", not config["pretrained"]),
                          ("--no-cache", not config["cache"]),
                          ("--smoke", config["smoke"])):
        if enabled:
            args.append(flag)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    scenario = args.scenario_dir
    log = Logger(scenario / "tuning.log")

    runs = [read_grid_run(scenario / f"lr_{lr}", lr) for lr in args.lrs]
    winner = choose(runs)
    summary = pd.DataFrame([{**{k: r[k] for k in ("lr", "best_epoch", "validation_macro_f1",
                                                  "validation_loss", "output_dir")},
                             "selected": i == winner} for i, r in enumerate(runs)])
    summary.to_csv(scenario / "tuning_summary.csv", index=False)
    for row in summary.to_dict("records"):
        log(json.dumps(row))
    chosen = runs[winner]
    log(f"Selected lr {chosen['lr']} (validation macro-F1={chosen['validation_macro_f1']:.6f} "
        f"at epoch {chosen['best_epoch']}); predicting test for this run only")

    # The test split is touched for the first time here, after the learning rate has been chosen.
    train_main(resume_args(chosen["config"], args.num_workers, args.device))

    source = Path(chosen["output_dir"])
    for name in COPIED_FILES:
        shutil.copy2(source / name, scenario / name)
    log(f"Copied {', '.join(COPIED_FILES)} from {source} to {scenario}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
