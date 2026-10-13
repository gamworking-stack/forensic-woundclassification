"""Run configuration for one training scenario (one model family x one input path)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


FAMILIES = (
    "mobilenet_v3_small",
    "vgg16",
    "resnet152",
    "efficientnet_b0",
    "googlenet",
    "vit_b16",
    "alexnet",
)
INPUT_PATHS = ("full", "roi")


@dataclass
class RunConfig:
    family: str
    input_path: str
    manifest: Path
    image_root: Path
    output_dir: Path
    roi_manifest: Path = Path("data/roi/roi_manifest.csv")
    full_scope: str = "matched"
    image_size: Optional[int] = None
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 25
    lr: float = 3e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    class_weights: bool = True
    seed: int = 42
    num_workers: int = 8
    device: str = "auto"
    pretrained: bool = True
    cache: bool = True
    resume: bool = False
    smoke: bool = False
    skip_test: bool = False
    cv_folds: Optional[Path] = None
    fold: Optional[int] = None
    checkpoint: str = "best"

    def to_json(self) -> Dict[str, Any]:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Fine-tune one model family on one input path (full image or ROI)."
    )
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--input-path", required=True, choices=INPUT_PATHS)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Split manifest (full) or ROI manifest (roi).")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Raw image tree (full) or ROI crop directory (roi).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, default=Path("data/roi/roi_manifest.csv"),
                        help="ROI manifest used to match full-image rows when --full-scope matched.")
    parser.add_argument("--full-scope", choices=("matched", "all"), default="matched",
                        help="Full path only: 'matched' keeps the images that also have an ROI crop.")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override the family's input size (tests only).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=None, help="Default 200, or 2 with --smoke.")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None, help="Default 8, or 0 with --smoke.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt if present.")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on 2 images per class per split.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Tuning run: write validation predictions only; never load the test split.")
    parser.add_argument("--cv-folds", type=Path, default=None,
                        help="CV run (T12b): fold table; test rows are dropped and --fold is validation.")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best",
                        help="Weights that predict: best validation epoch, or the last epoch (fixed-epoch CV).")
    args = parser.parse_args(argv)
    if (args.cv_folds is None) != (args.fold is None):
        parser.error("--cv-folds and --fold go together")
    if args.cv_folds is not None:
        args.skip_test = True  # CV never loads the test split

    if args.max_epochs is None:
        args.max_epochs = 2 if args.smoke else 200
    if args.num_workers is None:
        args.num_workers = 0 if args.smoke else 8
    return RunConfig(**vars(args))
