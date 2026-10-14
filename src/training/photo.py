"""Per-box crops grouped back into photos (T17): loss weights and photo-level predictions.

In the ``box`` input path every annotated box is its own crop, so one photo can give many rows
(up to 35). In the manifest a crop's photo is ``relative_path`` (the crop itself is
``roi_relative_path``; ``source_relative_path`` is only filled by live cropping runs, not by the
committed dry-run manifest). Prediction files name the photo in ``PHOTO_COLUMN``, because their
``relative_path`` is the crop. Training weights a crop by
1 / (crops of its photo) so every photo counts once; a photo's prediction is the mean of its
crops' probabilities.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


PHOTO_COLUMN = "source_relative_path"  # photo of each row in prediction files
MANIFEST_PHOTO_COLUMN = "relative_path"  # photo of each row in the ROI manifest
PHOTO_INPUT_PATHS = ("box",)


def manifest_photos(rows: pd.DataFrame) -> np.ndarray:
    """Each manifest row's source photo; every row must name one."""
    photos = rows[MANIFEST_PHOTO_COLUMN].astype(str).str.strip().to_numpy()
    if (photos == "").any():
        raise ValueError(f"{int((photos == '').sum())} manifest rows have an empty {MANIFEST_PHOTO_COLUMN!r}")
    return photos


def photo_weights(photos: Sequence[str]) -> np.ndarray:
    """1 / (rows sharing the photo) for every row, so each photo's rows sum to 1."""
    keys = pd.Series(list(photos))
    return (1.0 / keys.map(keys.value_counts())).to_numpy(dtype=float)


def photo_groups(photos: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """Photos in order of first appearance, and each row's index into that list."""
    codes, uniques = pd.factorize(pd.Series(list(photos)), sort=False)
    return list(uniques), codes


def photo_mean(photos: Sequence[str], labels: np.ndarray, probs: np.ndarray
               ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Mean crop probabilities per photo; every crop of a photo must have the photo's label."""
    names, codes = photo_groups(photos)
    labels = np.asarray(labels)
    photo_labels = np.full(len(names), -1)
    for i, code in enumerate(codes):
        if photo_labels[code] not in (-1, labels[i]):
            raise ValueError(f"Photo {names[code]!r} has crops with different labels")
        photo_labels[code] = labels[i]
    sums = np.zeros((len(names), probs.shape[1]))
    np.add.at(sums, codes, probs)
    counts = np.bincount(codes, minlength=len(names))[:, None]
    return names, photo_labels, sums / counts


def photo_prediction_frame(crops: pd.DataFrame) -> pd.DataFrame:
    """Photo-level prediction file from a crop-level one: same columns, one row per photo.

    ``relative_path`` becomes the photo; probabilities are the crop mean and the predicted class
    their argmax, so the result can be scored by the usual evaluation code.
    """
    if PHOTO_COLUMN not in crops.columns:
        raise ValueError(f"Crop predictions have no {PHOTO_COLUMN!r} column")
    prob_columns = [c for c in crops.columns if c.startswith("prob_")]
    classes = [c[len("prob_"):].replace("_", " ") for c in prob_columns]
    first = crops.groupby(PHOTO_COLUMN, sort=False).first()
    for column in ("pseudonimized_id", "split", "true_class"):
        varying = crops.groupby(PHOTO_COLUMN, sort=False)[column].nunique()
        if (varying > 1).any():
            raise ValueError(f"Photo {varying[varying > 1].index[0]!r} has crops with different {column}")
    mean = crops.groupby(PHOTO_COLUMN, sort=False)[prob_columns].mean()
    frame = pd.DataFrame({
        "relative_path": first.index.to_numpy(),
        "pseudonimized_id": first["pseudonimized_id"].to_numpy(),
        "split": first["split"].to_numpy(),
        "true_class": first["true_class"].to_numpy(),
        "predicted_class": [classes[i] for i in mean.to_numpy().argmax(axis=1)],
        "n_crops": crops.groupby(PHOTO_COLUMN, sort=False).size().to_numpy(),
    })
    for column in prob_columns:
        frame[column] = mean[column].to_numpy()
    return frame
