"""Manifest-driven image dataset shared by the full-image and ROI input paths."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pandas as pd
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset

from src.data.create_roi_dataset import find_source_image, verify_no_case_leakage
from src.training.config import RunConfig


ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLIT_NAMES = ("train", "validation", "test")
GROUP_COLUMN = "pseudonimized_id"
LABEL_COLUMN = "canonical_wound_label"
EXPECTED_CLASSES = 7


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_manifest(cfg: RunConfig) -> pd.DataFrame:
    """Read the scenario's manifest, apply the full-image scope, and check leakage and classes."""
    df = read_csv(cfg.manifest)
    df["split"] = df["split"].str.strip().str.lower()
    unknown = sorted(set(df["split"]) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"Unexpected split values in {cfg.manifest}: {unknown}")

    if cfg.input_path == "full" and cfg.full_scope == "matched":
        roi_paths = set(read_csv(cfg.roi_manifest)["relative_path"])
        df = df[df["relative_path"].isin(roi_paths)].reset_index(drop=True)

    verify_no_case_leakage(df.to_dict("records"), GROUP_COLUMN)
    if cfg.smoke:
        df = df.groupby(["split", LABEL_COLUMN], sort=False).head(2).reset_index(drop=True)
    check_class_coverage(df)
    return df


def class_names(df: pd.DataFrame) -> List[str]:
    return sorted(df[LABEL_COLUMN].unique())


def check_class_coverage(df: pd.DataFrame) -> None:
    classes = class_names(df)
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} classes, found {len(classes)}: {classes}")
    for split in SPLIT_NAMES:
        missing = set(classes) - set(df.loc[df["split"] == split, LABEL_COLUMN])
        if missing:
            raise ValueError(f"Split {split!r} has no images of: {sorted(missing)}")


def resolve_image_paths(df: pd.DataFrame, input_path: str, image_root: Path) -> List[Path]:
    """Resolve every row to an image file, failing with the full list of missing rows."""
    paths: List[Path] = []
    missing: List[str] = []
    for row in df.to_dict("records"):
        if input_path == "roi":
            candidate: Optional[Path] = image_root / row["roi_relative_path"]
            if not candidate.is_file():
                candidate = None
        else:
            candidate = find_source_image(image_root, row)
        if candidate is None:
            missing.append(row.get("roi_relative_path") if input_path == "roi" else row["relative_path"])
        else:
            paths.append(candidate)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found under {image_root}: {missing}")
    return paths


def load_image(path: Path, max_side: Optional[int] = None) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if max_side is not None:
        image.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return image


class ManifestImageDataset(Dataset):
    """Images and integer labels for one split.

    With ``cache_side`` set, every image is decoded once up front and kept in memory with its
    longest side shrunk to ``cache_side``; the raw photos are up to 6000 x 4000.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        transform: Callable,
        cache_side: Optional[int] = None,
        threads: int = 8,
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
        self.cache: Optional[List[Image.Image]] = None
        if cache_side is not None:
            with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
                self.cache = list(pool.map(lambda p: load_image(p, cache_side), self.paths))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = self.cache[idx] if self.cache is not None else load_image(self.paths[idx])
        return self.transform(image), self.labels[idx]
