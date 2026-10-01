"""Train/predict loop, early stopping, seeding and checkpoints shared by every family."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive NumPy and Python seeds from the worker's torch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def capture_rng() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    """Write atomically so a job killed mid-save never leaves a truncated checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    # Our own files; they hold RNG states, which weights_only=True rejects.
    return torch.load(path, map_location=device, weights_only=False)


@dataclass
class EarlyStopping:
    """Track the best validation macro-F1 (ties broken by lower loss) and stop after ``patience``."""

    patience: int
    best_f1: float = float("-inf")
    best_loss: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0
    stopped: bool = False

    def step(self, epoch: int, macro_f1: float, loss: float) -> bool:
        """Record one epoch; return True when it is the new best."""
        improved = macro_f1 > self.best_f1 or (macro_f1 == self.best_f1 and loss < self.best_loss)
        if improved:
            self.best_f1, self.best_loss, self.best_epoch, self.bad_epochs = macro_f1, loss, epoch, 0
        else:
            self.bad_epochs += 1
            self.stopped = self.bad_epochs >= self.patience
        return improved

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(num_classes)),
                                   average="macro", zero_division=0)),
    }


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """One pass over the training loader; returns (mean loss, labels, predicted labels)."""
    model.train()
    total_loss, count = 0.0, 0
    labels, preds = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = model(images)
        loss = criterion(logits.float(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        count += len(targets)
        labels.append(targets.cpu())
        preds.append(logits.argmax(dim=1).cpu())
    return total_loss / count, torch.cat(labels).numpy(), torch.cat(preds).numpy()


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate without gradients; returns (mean loss, labels, class probabilities)."""
    model.eval()
    total_loss, count = 0.0, 0
    labels, probs = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device):
            logits = model(images)
        logits = logits.float()
        total_loss += criterion(logits, targets).item() * len(targets)
        count += len(targets)
        labels.append(targets.cpu())
        probs.append(torch.softmax(logits, dim=1).cpu())
    return total_loss / count, torch.cat(labels).numpy(), torch.cat(probs).numpy()
