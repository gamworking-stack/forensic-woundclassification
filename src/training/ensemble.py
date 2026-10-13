#!/usr/bin/env python3
"""Soft-voting ensemble: average the class probabilities of finished train.py runs.

Members are ordinary train.py runs (AlexNet with seeds 43-45 for S15/S16). Every member must
have the same rows in the same order and the same class columns; any mismatch is an error,
never a silent re-sort. Weights are equal, so nothing is tuned and the combination does not
use validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.common import Logger, environment_info, git_state  # noqa: E402
from src.training.engine import load_checkpoint  # noqa: E402


ID_COLUMNS = ["relative_path", "pseudonimized_id", "split", "true_class"]
PREDICTION_SPLITS = ("validation", "test")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft-vote finished train.py runs into one ensemble.")
    parser.add_argument("--members", type=Path, nargs="+", required=True,
                        help="Output folders of finished train.py runs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=PREDICTION_SPLITS, default=list(PREDICTION_SPLITS),
                        help="Prediction files to combine; CV folds (T12b) have validation only.")
    return parser.parse_args(argv)


def read_member(member_dir: Path, splits=PREDICTION_SPLITS) -> Dict[str, Any]:
    """Load a member's config, best-checkpoint summary and prediction files."""
    predictions = {}
    for split in splits:
        path = member_dir / f"{split}_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"{member_dir} has no {path.name}; has its training run finished?")
        predictions[split] = pd.read_csv(path, dtype={c: str for c in ID_COLUMNS + ["predicted_class"]})
    config = json.loads((member_dir / "run_config.json").read_text(encoding="utf-8"))
    best = load_checkpoint(member_dir / "best.pt", torch.device("cpu"))
    return {
        "dir": member_dir,
        "config": config,
        "best_epoch": int(best["epoch"]),
        "best_validation_macro_f1": float(best["validation_macro_f1"]),
        "predictions": predictions,
    }


def check_members(members: List[Dict[str, Any]]) -> List[str]:
    """Members must share input path and classes and have distinct seeds; return the class list."""
    first = members[0]["config"]
    for member in members[1:]:
        config = member["config"]
        for key in ("input_path", "classes"):
            if config[key] != first[key]:
                raise ValueError(f"{member['dir']} has {key}={config[key]!r}, "
                                 f"but {members[0]['dir']} has {first[key]!r}")
    seeds = [m["config"]["seed"] for m in members]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Ensemble members must have distinct seeds, got {seeds}")
    return list(first["classes"])


def soft_vote(frames: List[pd.DataFrame], names: List[str], classes: List[str]) -> pd.DataFrame:
    """Mean of the members' probabilities per image and class; predicted class is the argmax."""
    prob_columns = [f"prob_{c.replace(' ', '_')}" for c in classes]
    reference = frames[0]
    for frame, name in zip(frames, names):
        if [c for c in frame.columns if c.startswith("prob_")] != prob_columns:
            raise ValueError(f"{name}: probability columns do not match classes {classes}")
        if len(frame) != len(reference) or not frame[ID_COLUMNS].equals(reference[ID_COLUMNS]):
            raise ValueError(f"{name}: rows differ from {names[0]} (different images or order)")

    mean = np.mean([frame[prob_columns].to_numpy() for frame in frames], axis=0)
    voted = reference[ID_COLUMNS].copy()
    voted["predicted_class"] = [classes[i] for i in mean.argmax(axis=1)]
    for j, column in enumerate(prob_columns):
        voted[column] = mean[:, j]
    return voted


def representative_index(members: List[Dict[str, Any]]) -> int:
    """Member for Grad-CAM (T14): highest validation macro-F1, ties to the lower seed."""
    return max(range(len(members)),
               key=lambda i: (members[i]["best_validation_macro_f1"], -members[i]["config"]["seed"]))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "ensemble.log")

    members = [read_member(d, args.splits) for d in args.members]
    classes = check_members(members)
    names = [str(m["dir"]) for m in members]

    for split in args.splits:
        voted = soft_vote([m["predictions"][split] for m in members], names, classes)
        voted.to_csv(out / f"{split}_predictions.csv", index=False)

    rep = representative_index(members)
    table = pd.DataFrame([{
        "member": i + 1,
        "seed": m["config"]["seed"],
        "best_epoch": m["best_epoch"],
        "best_validation_macro_f1": m["best_validation_macro_f1"],
        "output_dir": str(m["dir"]),
        "representative": i == rep,
    } for i, m in enumerate(members)])
    table.to_csv(out / "ensemble_members.csv", index=False)

    run_config = {
        "family": "ensemble",
        "member_family": sorted({m["config"]["family"] for m in members}),
        "input_path": members[0]["config"]["input_path"],
        "members": names,
        "seeds": [m["config"]["seed"] for m in members],
        "combination": "mean",
        "representative_member": rep + 1,
        "classes": classes,
        **git_state(),
        "environment": environment_info(torch.device("cpu")),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (out / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    for row in table.to_dict("records"):
        log(json.dumps(row))
    log(f"Soft-voted {len(members)} members into {out}; representative member {rep + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
