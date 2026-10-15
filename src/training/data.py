"""Manifest-driven image dataset shared by the full-image and ROI input paths."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset

from src.data.create_roi_dataset import find_source_image, verify_no_case_leakage


ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLIT_NAMES = ("train", "validation", "test")
GROUP_COLUMN = "pseudonimized_id"
LABEL_COLUMN = "canonical_wound_label"
EXPECTED_CLASSES = 7


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_manifest(
    manifest: Path,
    input_path: str,
    full_scope: str = "matched",
    roi_manifest: Path = Path("data/roi/roi_manifest.csv"),
    smoke: bool = False,
    cv_folds: Optional[Path] = None,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    """Read a scenario's manifest, apply the full-image scope, and check leakage and classes.

    With ``cv_folds`` and ``fold`` (T12b) the test rows are dropped, the cases of ``fold`` become
    the validation split and every other development case the train split.
    """
    df = read_csv(manifest)
    df["split"] = df["split"].str.strip().str.lower()
    unknown = sorted(set(df["split"]) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"Unexpected split values in {manifest}: {unknown}")
    splits = SPLIT_NAMES
    if cv_folds is not None:
        df = cv_split(df, cv_folds, fold)
        splits = ("train", "validation")

    if input_path == "full" and full_scope == "matched":
        roi_paths = set(read_csv(roi_manifest)["relative_path"])
        df = df[df["relative_path"].isin(roi_paths)].reset_index(drop=True)

    verify_no_case_leakage(df.to_dict("records"), GROUP_COLUMN)
    if smoke:
        df = df.groupby(["split", LABEL_COLUMN], sort=False).head(2).reset_index(drop=True)
    check_class_coverage(df, splits)
    return df


def cv_split(df: pd.DataFrame, cv_folds: Path, fold: Optional[int]) -> pd.DataFrame:
    """Development rows only; ``fold`` becomes validation, the other folds train."""
    folds = read_csv(cv_folds).set_index(GROUP_COLUMN)["fold"].astype(int)
    if fold not in set(folds):
        raise ValueError(f"Fold {fold} is not in {cv_folds} (folds {sorted(set(folds))})")
    df = df[df["split"] != "test"].copy()
    assigned = df[GROUP_COLUMN].map(folds)
    if assigned.isna().any():
        missing = sorted(set(df.loc[assigned.isna(), GROUP_COLUMN]))
        raise ValueError(f"{len(missing)} development cases have no fold in {cv_folds}: {missing[:5]}")
    df["split"] = ["validation" if f == fold else "train" for f in assigned]
    return df.reset_index(drop=True)


def class_names(df: pd.DataFrame) -> List[str]:
    return sorted(df[LABEL_COLUMN].unique())


def check_class_coverage(df: pd.DataFrame, splits: Sequence[str] = SPLIT_NAMES) -> None:
    classes = class_names(df)
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} classes, found {len(classes)}: {classes}")
    for split in splits:
        missing = set(classes) - set(df.loc[df["split"] == split, LABEL_COLUMN])
        if missing:
            raise ValueError(f"Split {split!r} has no images of: {sorted(missing)}")


CROP_INPUT_PATHS = ("roi", "box", "crop")  # images read from roi_relative_path


def resolve_image_paths(df: pd.DataFrame, input_path: str, image_root: Path) -> List[Path]:
    """Resolve every row to an image file, failing with the full list of missing rows."""
    paths: List[Path] = []
    missing: List[str] = []
    for row in df.to_dict("records"):
        if input_path in CROP_INPUT_PATHS:
            candidate: Optional[Path] = image_root / row["roi_relative_path"]
            if not candidate.is_file():
                candidate = None
        else:
            candidate = find_source_image(image_root, row)
        if candidate is None:
            missing.append(row.get("roi_relative_path") if input_path in CROP_INPUT_PATHS else row["relative_path"])
        else:
            paths.append(candidate)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found under {image_root}: {missing}")
    return paths


def prepare_rows(df: pd.DataFrame, input_path: str, image_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Add resolved image paths and integer labels; return the rows and the class list."""
    classes = class_names(df)
    paths = resolve_image_paths(df, input_path, image_root)
    df = df.assign(
        image_path=paths,
        image_relative_path=[p.relative_to(image_root).as_posix() for p in paths],
        label_index=df[LABEL_COLUMN].map({name: i for i, name in enumerate(classes)}),
    )
    return df, classes


def load_image(path: Path, max_side: Optional[int] = None) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if max_side is not None:
        image.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return image


class ManifestImageDataset(Dataset):
    """Images and integer labels for one split.

    With ``cache_side`` set, every image is decoded once up front and kept in memory with its
    longest side shrunk to ``cache_side``; the raw photos are up to 6000 x 4000. With ``weights``
    (T17 per-box training) each item also returns its loss weight.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        transform: Callable,
        cache_side: Optional[int] = None,
        threads: int = 8,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.weights = None if weights is None else [float(w) for w in weights]
        self.transform = transform
        self.cache: Optional[List[Image.Image]] = None
        if cache_side is not None:
            with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
                self.cache = list(pool.map(lambda p: load_image(p, cache_side), self.paths))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = self.cache[idx] if self.cache is not None else load_image(self.paths[idx])
        if self.weights is not None:
            return self.transform(image), self.labels[idx], self.weights[idx]
        return self.transform(image), self.labels[idx]
