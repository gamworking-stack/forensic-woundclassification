#!/usr/bin/env python3
"""Manual reference boxes and the box rules shared by every detector arm (T19 §2.1, §2.3).

The reference boxes are the **unpadded** CVAT boxes: the ROI manifests only hold padded crop
coordinates. ``read_manual_boxes`` recovers them from the CVAT YOLO exports and refuses to return
anything unless every box, padded the way T03 pads it, reproduces its row of the per-box manifest
(``data/roi_box/roi_manifest.csv``) exactly. Boxes are ``x1, y1, x2, y2`` floats in source pixels
of the EXIF-oriented photo, as in the manifests.

The helpers implement T19 §2.3 for any detector's output on one photo:

1. ``candidate_boxes``: drop oversize boxes, NMS, keep the top ``TOP_K`` by score (mAP is computed here);
2. ``final_boxes``: operating threshold, per-photo cap, then the fallback (top box, else full image);
3. ``padded_crop_box``: the 8%-padded crop around boxes, with T03's rounding and clipping.

``greedy_match`` is the one-to-one matching used for the F1 threshold and operating-point metrics.

Run as a script to write the frozen reference: ``reference_boxes.csv`` and ``reference_limits.json``
(the oversize share and box cap of T19 §2.3, from the training split).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_roi_dataset import (  # noqa: E402
    compute_union_box,
    manifest_key,
    parse_export,
    resolve_export_files,
)
from src.training.data import GROUP_COLUMN, LABEL_COLUMN  # noqa: E402
from src.training.photo import MANIFEST_PHOTO_COLUMN  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402


PADDING = 0.08
NMS_IOU = 0.5
MATCH_IOU = 0.5
TOP_K = 100
FALLBACKS = ("none", "top_box", "full_image")
COORDINATES = ["x1", "y1", "x2", "y2"]
REFERENCE_COLUMNS = [MANIFEST_PHOTO_COLUMN, GROUP_COLUMN, "split", LABEL_COLUMN, "box_index",
                     *COORDINATES, "source_width", "source_height", "annotation_export"]


# ---------- manual reference boxes ----------

def read_manual_boxes(exports: Sequence[Path], box_manifest: Path) -> pd.DataFrame:
    """Unpadded CVAT boxes for every photo of the per-box manifest, checked crop by crop.

    Raises unless every manifest photo is annotated, has as many boxes as crops, and each box padded
    by the manifest's ``padding_ratio`` gives exactly that crop's ``x1..y2``. Annotations of photos
    outside the manifest (the 4 unmatched in T17) are skipped.
    """
    # Read the per-box manifest and check it uses one padding ratio for every crop
    manifest = pd.read_csv(box_manifest, dtype=str, keep_default_na=False)
    paddings = set(manifest["padding_ratio"].astype(float))
    if len(paddings) != 1:
        raise ValueError(f"Manifest mixes padding ratios: {sorted(paddings)}")
    padding = paddings.pop()

    # Group the manifest crops by photo (wound type, file stem), the key CVAT annotations use
    crops_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in manifest.to_dict("records"):
        crops_by_key[manifest_key(row)].append(row)

    # Walk every CVAT export (duplicate downloads skipped) and every annotated photo in it
    rows: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], str] = {}
    for export in resolve_export_files(list(exports)):
        _, annotations, invalid = parse_export(export)
        if invalid:
            raise ValueError(f"{export.name}: {len(invalid)} invalid annotation lines, e.g. {invalid[0]}")
        for stem, labeled_boxes in sorted(annotations.items()):
            # One photo's boxes, grouped by label (a CVAT task has one label)
            boxes_by_label: Dict[str, List[List[float]]] = defaultdict(list)
            for label, box in labeled_boxes:
                boxes_by_label[label].append(box)
            for label, boxes in boxes_by_label.items():
                # A photo may be annotated in only one export; photos outside the manifest are skipped
                key = (label, stem)
                if key in seen:
                    raise ValueError(f"Annotation collision for {key}: {seen[key]} and {export.name}")
                seen[key] = export.name
                if key not in crops_by_key:
                    continue
                rows += _check_photo(key, boxes, crops_by_key[key], padding, export.name)

    # Every manifest photo must have been annotated; return the boxes sorted by split and photo
    missing = sorted(set(crops_by_key) - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} manifest photos have no annotation, e.g. {missing[:3]}")
    frame = pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
    return frame.sort_values(["split", MANIFEST_PHOTO_COLUMN, "box_index"], kind="stable").reset_index(drop=True)


def _check_photo(key: Tuple[str, str], boxes: List[List[float]], crops: List[Dict[str, Any]],
                 padding: float, export: str) -> List[Dict[str, Any]]:
    """Reference rows for one photo; its boxes must reproduce its crops in ``crop_index`` order."""
    # Crops in crop_index order, the same order as the boxes in the CVAT file
    crops = sorted(crops, key=lambda c: int(c["crop_index"]))
    if len(boxes) != len(crops):
        raise ValueError(f"{key}: {len(boxes)} CVAT boxes but {len(crops)} manifest crops")
    rows = []
    # Pad each box exactly as T03 did and require it to equal its manifest crop
    for index, (box, crop) in enumerate(zip(boxes, crops), start=1):
        if int(crop["crop_index"]) != index:
            raise ValueError(f"{key}: crop indices are not 1..{len(crops)}")
        width, height = int(crop["source_width"]), int(crop["source_height"])
        padded = compute_union_box([box], width, height, padding)
        expected = tuple(int(crop[c]) for c in COORDINATES)
        if padded != expected:
            raise ValueError(f"{key} box {index}: padded CVAT box {padded} != manifest crop {expected}")
        # Convert the YOLO box (centre, size, 0-1) to pixel corners, clipped to the image
        cx, cy, bw, bh = box
        rows.append({
            MANIFEST_PHOTO_COLUMN: crop[MANIFEST_PHOTO_COLUMN], GROUP_COLUMN: crop[GROUP_COLUMN],
            "split": crop["split"].strip().lower(), LABEL_COLUMN: crop[LABEL_COLUMN], "box_index": index,
            "x1": max(0.0, (cx - bw / 2) * width), "y1": max(0.0, (cy - bh / 2) * height),
            "x2": min(float(width), (cx + bw / 2) * width), "y2": min(float(height), (cy + bh / 2) * height),
            "source_width": width, "source_height": height, "annotation_export": export,
        })
    return rows


def training_limits(reference: pd.DataFrame) -> Dict[str, Any]:
    """The oversize share, the per-photo box cap (T19 §2.3) and the box-size tertiles, from the training split only."""
    # Only training boxes set the limits, so validation and test never influence them
    train = reference[reference["split"] == "train"]
    if train.empty:
        raise ValueError("Reference boxes have no training split")
    # Each training box's area as a share of its photo
    shares = area_share(train[COORDINATES].to_numpy(float), train["source_width"].to_numpy(float),
                        train["source_height"].to_numpy(float))
    return {
        "max_box_share": float(shares.max()),
        # Size bins for AP by box size: COCO's pixel bins put every box of a 4000 px photo in "large".
        "size_bins": [float(np.quantile(shares, 1 / 3)), float(np.quantile(shares, 2 / 3))],
        "max_boxes_per_photo": int(train.groupby(MANIFEST_PHOTO_COLUMN).size().max()),
        "train_photos": int(train[MANIFEST_PHOTO_COLUMN].nunique()),
        "train_boxes": int(len(train)),
    }


# ---------- box helpers (one photo at a time) ----------

def as_boxes(boxes: Any) -> np.ndarray:
    """``(N, 4)`` float array of ``x1, y1, x2, y2``; an empty input gives shape ``(0, 4)``."""
    # Reshape to rows of x1, y1, x2, y2 and reject boxes with x2 < x1 or y2 < y1
    array = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if (array[:, 2] < array[:, 0]).any() or (array[:, 3] < array[:, 1]).any():
        raise ValueError("Boxes need x2 >= x1 and y2 >= y1")
    return array


def box_area(boxes: Any) -> np.ndarray:
    b = as_boxes(boxes)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def area_share(boxes: Any, width: Any, height: Any) -> np.ndarray:
    """Each box's area as a share of its image's area."""
    return box_area(boxes) / (np.asarray(width, dtype=float) * np.asarray(height, dtype=float))


def iou_matrix(a: Any, b: Any) -> np.ndarray:
    """Intersection over union of every box in ``a`` with every box in ``b``: shape ``(len(a), len(b))``."""
    a, b = as_boxes(a), as_boxes(b)
    # Corners of the overlap of every box in a with every box in b
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    # Overlap area (0 when boxes do not touch), then IoU = overlap / union
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def score_order(scores: Any) -> np.ndarray:
    """Indices by descending score; equal scores keep their input order, so results are deterministic."""
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def nms(boxes: Any, scores: Any, iou_threshold: float = NMS_IOU) -> np.ndarray:
    """Greedy non-maximum suppression; kept indices in descending score order.

    A box is suppressed when its IoU with an already kept box is strictly above ``iou_threshold``.
    """
    boxes = as_boxes(boxes)
    # Take the best remaining box, keep it, and drop every box overlapping it by more than the threshold
    remaining = list(score_order(scores))
    keep: List[int] = []
    while remaining:
        best = remaining.pop(0)
        keep.append(best)
        if remaining:
            overlaps = iou_matrix(boxes[[best]], boxes[remaining])[0]
            remaining = [i for i, o in zip(remaining, overlaps) if o <= iou_threshold]
    return np.asarray(keep, dtype=int)


def candidate_boxes(boxes: Any, scores: Any, width: float, height: float, max_share: float, *,
                    iou_threshold: float = NMS_IOU, top_k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
    """T19 §2.3 step 1: drop boxes larger than ``max_share`` of the image, NMS, keep the top ``top_k``."""
    boxes, scores = as_boxes(boxes), np.asarray(scores, dtype=float).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError(f"{len(boxes)} boxes but {len(scores)} scores")
    # Drop boxes bigger than the largest manual training box, then NMS, then the top boxes
    small = area_share(boxes, width, height) <= max_share
    boxes, scores = boxes[small], scores[small]
    keep = nms(boxes, scores, iou_threshold)[:top_k]
    return boxes[keep], scores[keep]


def final_boxes(boxes: Any, scores: Any, width: float, height: float, threshold: float,
                max_boxes: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """T19 §2.3 step 3 on a photo's candidate boxes: threshold, cap, then the fallback.

    Boxes with ``score >= threshold`` are kept, at most ``max_boxes`` by score. With none left, the
    top-scoring candidate is kept (``"top_box"``); with no candidate at all, the full image is the box
    and its score is NaN (``"full_image"``). Otherwise the fallback is ``"none"``.
    """
    boxes, scores = as_boxes(boxes), np.asarray(scores, dtype=float).reshape(-1)
    order = score_order(scores)
    # Boxes at or above the threshold, highest first, at most max_boxes
    kept = [i for i in order if scores[i] >= threshold][:max_boxes]
    if kept:
        return boxes[kept], scores[kept], "none"
    # Nothing passed the threshold: keep the single best box
    if len(order):
        return boxes[order[:1]], scores[order[:1]], "top_box"
    # No box at all: the whole photo becomes the crop
    return as_boxes([0.0, 0.0, float(width), float(height)]), np.array([np.nan]), "full_image"


def padded_crop_box(boxes: Any, width: int, height: int, padding: float = PADDING) -> Tuple[int, int, int, int]:
    """The crop around the union of pixel ``boxes``, padded and rounded exactly as T03 does."""
    b = as_boxes(boxes)
    if not len(b):
        raise ValueError("A crop needs at least one box")
    # Union of all boxes, then 8% of its width and height added on each side, clipped to the photo
    x1, y1 = b[:, 0].min(), b[:, 1].min()
    x2, y2 = b[:, 2].max(), b[:, 3].max()
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, int(round(x1 - pad_x))), max(0, int(round(y1 - pad_y))),
            min(int(width), int(round(x2 + pad_x))), min(int(height), int(round(y2 + pad_y))))


def greedy_match(pred_boxes: Any, pred_scores: Any, true_boxes: Any,
                 iou_threshold: float = MATCH_IOU) -> List[Tuple[int, int, float]]:
    """One-to-one matching as COCO does it: predictions by descending score, each taking the unmatched
    reference box it overlaps most, if that IoU is at least ``iou_threshold``.

    Returns ``(prediction index, reference index, IoU)`` for every match.
    """
    pred_boxes, true_boxes = as_boxes(pred_boxes), as_boxes(true_boxes)
    if not len(pred_boxes) or not len(true_boxes):
        return []
    # Highest-scoring prediction first; each takes the free manual box it overlaps most, if IoU >= threshold
    overlaps = iou_matrix(pred_boxes, true_boxes)
    taken = np.zeros(len(true_boxes), dtype=bool)
    matches = []
    for p in score_order(pred_scores):
        candidates = np.where(~taken, overlaps[p], -1.0)
        t = int(np.argmax(candidates))
        if candidates[t] >= iou_threshold:
            taken[t] = True
            matches.append((int(p), t, float(overlaps[p, t])))
    return matches


# ---------- script: write the frozen reference ----------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the unpadded manual reference boxes (T19 §2.1).")
    parser.add_argument("--exports", type=Path, nargs="+", required=True,
                        help="The CVAT .zip exports (30 Sep 2026; 23 Oct 2026 for T23), or a folder holding them.")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "roi_auto")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    # Read and check the manual boxes, compute the training limits
    reference = read_manual_boxes(args.exports, args.box_manifest)
    limits = training_limits(reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Write the reference boxes and a JSON with the limits and counts
    reference.to_csv(args.output_dir / "reference_boxes.csv", index=False, float_format="%.10g")
    summary = {
        **limits,
        "boxes_by_split": reference["split"].value_counts().sort_index().to_dict(),
        "photos_by_split": reference.groupby("split")[MANIFEST_PHOTO_COLUMN].nunique().to_dict(),
        "box_manifest": args.box_manifest.as_posix(),
        "annotation_exports": sorted(reference["annotation_export"].unique()),
        "checked": "every box padded by the manifest padding_ratio equals its per-box manifest crop",
    }
    (args.output_dir / "reference_limits.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{len(reference)} reference boxes on {reference[MANIFEST_PHOTO_COLUMN].nunique()} photos; "
          f"training limits: max box share {limits['max_box_share']:.4f}, "
          f"max boxes per photo {limits['max_boxes_per_photo']}. Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
