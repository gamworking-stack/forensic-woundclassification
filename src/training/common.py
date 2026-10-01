"""Logging, device choice, run provenance and prediction files shared by every entry point."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchvision

from src.training.data import GROUP_COLUMN


REPO_ROOT = Path(__file__).resolve().parents[2]


class Logger:
    """Print to stdout and append to a log file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def git_state() -> Dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


def environment_info(device: torch.device) -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def prediction_frame(
    rows: pd.DataFrame, labels: np.ndarray, probs: np.ndarray, classes: List[str]
) -> pd.DataFrame:
    """One row per image: identifiers, true and predicted class, and one probability per class."""
    frame = pd.DataFrame({
        "relative_path": rows["image_relative_path"].to_numpy(),
        "pseudonimized_id": rows[GROUP_COLUMN].to_numpy(),
        "split": rows["split"].to_numpy(),
        "true_class": [classes[i] for i in labels],
        "predicted_class": [classes[i] for i in probs.argmax(axis=1)],
    })
    for j, name in enumerate(classes):
        frame[f"prob_{name.replace(' ', '_')}"] = probs[:, j]
    return frame
