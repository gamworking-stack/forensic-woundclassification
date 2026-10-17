"""Photos, images, detections files and the test guard shared by the detector arms (T19 §3.2).

Every detector writes one detections file per configuration: one row per raw box with the photo
(``relative_path``), ``score`` and ``x1, y1, x2, y2`` in source pixels of the EXIF-oriented photo
(the frame of the manual boxes). Detectors run on the development photos first; the test photos
are run only for an arm whose ``selection.json`` exists and passed its gate, with the chosen
configuration (``test_allowed``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.data.create_roi_dataset import find_source_image
from src.detection.boxes import COORDINATES, as_boxes
from src.training.data import GROUP_COLUMN, LABEL_COLUMN
from src.training.photo import MANIFEST_PHOTO_COLUMN


PHOTO = MANIFEST_PHOTO_COLUMN
SPLITS = ("train", "validation", "test")
DETECTION_COLUMNS = [PHOTO, "score", *COORDINATES]
PHOTO_COLUMNS = [GROUP_COLUMN, "wound_id", "filename", "wound_label", LABEL_COLUMN, PHOTO, "split"]


def photo_table(box_manifest: Path, splits: Sequence[str] = SPLITS) -> pd.DataFrame:
    """One row per photo of ``splits`` from the per-box manifest, with its source size."""
    # The per-box manifest has one row per crop; keep the first row of each photo
    crops = pd.read_csv(box_manifest, dtype=str, keep_default_na=False)
    crops["split"] = crops["split"].str.strip().str.lower()
    photos = crops.drop_duplicates(PHOTO)[[*PHOTO_COLUMNS, "source_width", "source_height"]]
    # Only the requested splits, in a fixed photo order
    photos = photos[photos["split"].isin(splits)].sort_values(PHOTO).reset_index(drop=True)
    return photos.astype({"source_width": int, "source_height": int})


def load_photo(image_root: Path, row: Dict, check_size: bool = True) -> Image.Image:
    """The EXIF-oriented RGB photo of a manifest row; its size must match the manifest's."""
    # Find the raw photo file the same way the ROI cropping code does
    path = find_source_image(image_root, row)
    if path is None:
        raise FileNotFoundError(f"No source image for {row[PHOTO]} under {image_root}")
    # Rotate it upright (EXIF) as the manual boxes assume, and check its size
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if check_size and image.size != (int(row["source_width"]), int(row["source_height"])):
        raise ValueError(f"{row[PHOTO]}: image is {image.size}, manifest says "
                         f"{(row['source_width'], row['source_height'])}")
    return image


def detection_rows(photo: str, boxes: np.ndarray, scores: np.ndarray, width: int, height: int,
                   extra: Optional[Dict[str, Sequence]] = None) -> List[Dict]:
    """Rows of a detections file, boxes clipped to the image and empty boxes dropped."""
    # Clip boxes to the photo and drop any that became empty
    boxes = as_boxes(np.clip(np.asarray(boxes, dtype=float).reshape(-1, 4), 0, [width, height, width, height]))
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    rows = []
    # One row per box: photo, score, corners, plus any extra column (e.g. 4P's source class)
    for i in np.flatnonzero(keep):
        row = {PHOTO: photo, "score": float(scores[i]), **dict(zip(COORDINATES, boxes[i].tolist()))}
        for name, values in (extra or {}).items():
            row[name] = values[i]
        rows.append(row)
    return rows


def write_detections(rows: Iterable[Dict], path: Path, extra_columns: Sequence[str] = ()) -> None:
    """Write a detections file; never over an existing one (selection.json records their hashes)."""
    # Never overwrite: selection.json stores each detections file's hash
    if path.exists():
        raise FileExistsError(f"{path} exists; detections files are written once (write test runs to their own folder)")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows), columns=[*DETECTION_COLUMNS, *extra_columns]).to_csv(path, index=False, float_format="%.6g")


def test_allowed(selection: Optional[Path], config: str) -> None:
    """Raise unless ``selection`` is the arm's frozen selection, passed its gate, and chose ``config``."""
    # Test is allowed only after a frozen selection that passed its gate and chose this configuration
    if selection is None or not selection.is_file():
        raise SystemExit("Test photos are run only after the arm's selection.json exists (T19 §2.6); pass --selection")
    chosen = json.loads(selection.read_text(encoding="utf-8"))
    if not chosen.get("gate_passed", False):
        raise SystemExit(f"{selection}: the arm failed its recall gate and never touches test (T19 §2.4)")
    if chosen.get("chosen_config") != config:
        raise SystemExit(f"{selection} chose {chosen.get('chosen_config')!r}, not {config!r}")
