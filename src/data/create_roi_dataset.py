#!/usr/bin/env python3
"""Create a leakage-safe ROI classification dataset from CVAT exports.

Parses CVAT YOLO 1.1 (or CVAT for images 1.1 XML) annotation archives, matches annotations with the authoritative
case-level split manifest, computes padded union bounding boxes, and crops the visible
wound regions. Each ROI sample strictly inherits the case identifier and split partition
(train, validation, test) from its source image.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from PIL import Image, ImageFile, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.validate_metadata import LABEL_NORMALIZATION_MAP  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLIT_NAMES = ("train", "validation", "test")

BROWSER_COPY_SUFFIX = re.compile(r" \(\d+\)$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}



def normalize_label(val: str) -> str:
    """Normalize raw label string to canonical form."""
    clean = re.sub(r"\s+", " ", str(val).strip().lower())
    return LABEL_NORMALIZATION_MAP.get(clean, clean)


def safe_folder(val: str) -> str:
    """Clean label string for folder creation."""
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", str(val)).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create wound ROI crop dataset preserving case-level split."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_DIR / "splits/case_level_split_manifest.csv",
        help="Path to authoritative case-level split manifest.",
    )
    parser.add_argument(
        "--exports",
        type=Path,
        nargs="+",
        required=True,
        help="CVAT YOLO 1.1 or CVAT for images 1.1 annotation .zip files, or a directory containing them.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory containing raw image files.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path("../04_Early_experiments/Researcher_ROI_results/dataset_information/roi_manifest.csv"),
        help="Optional reference manifest to load image dimensions if local binaries are absent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "roi",
        help="Directory to save roi_manifest.csv, crop_summary.json, and cropped images.",
    )
    parser.add_argument(
        "--group-column",
        default="pseudonimized_id",
        help="Grouping column for cases.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Symmetric padding ratio around bounding box (default 0.08 = 8%).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=8,
        help="Minimum crop dimension in pixels (default 8).",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("union", "individual"),
        default="union",
        help="Crop mode: 'union' combines multiple bboxes into one crop per image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: parses annotations and builds manifest without writing image files.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear output directory before processing.",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path, group_col: str) -> List[Dict[str, Any]]:
    """Read the authoritative split manifest, validating columns, splits, and case leakage."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Read every manifest row as a dict of strings
    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Check the required columns, then that every split name is known
    required = {group_col, "split", "filename"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(f"Manifest missing required columns: {required.difference(rows[0].keys() if rows else set())}")

    for row in rows:
        if row["split"].strip().lower() not in SPLIT_NAMES:
            raise ValueError(f"Unexpected split value: {row['split']!r}")

    # Refuse a manifest where a case spans two splits
    verify_no_case_leakage(rows, group_col)
    return rows


def verify_no_case_leakage(rows: List[Dict[str, Any]], group_col: str) -> None:
    """Raise if any case appears in more than one split."""
    # Collect the case IDs of each split
    groups_by_split: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        groups_by_split[str(row["split"]).strip().lower()].add(str(row[group_col]))
    # Every pair of splits must share no case
    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1:]:
            shared = groups_by_split[first] & groups_by_split[second]
            if shared:
                raise AssertionError(f"Case leakage between {first} and {second}: {sorted(shared)}")


def class_label(row: Dict[str, Any]) -> str:
    """The row's canonical training label."""
    return normalize_label(row.get("canonical_wound_label") or row.get("wound_label", ""))


def manifest_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """Join key between manifest rows and annotations: (canonical label, filename stem).

    A photo relabelled after annotation (T23) keeps its old raw label in ``source_label``: its CVAT
    boxes and its raw image folder are still under that label, so the key uses it.
    """
    label = normalize_label(row["source_label"]) if row.get("source_label") else class_label(row)
    return label, Path(str(row["filename"]).strip()).stem


def parse_yolo_export(
    export_path: Path,
) -> Tuple[List[str], Dict[str, List[Tuple[str, List[float]]]], List[Dict[str, Any]]]:
    """Parse YOLO annotations from a CVAT zip export.

    Returns the normalized class names, the valid boxes per image stem, and every rejected line
    with the reason it was rejected.
    """
    # Valid boxes per image stem, and the rejected lines with their reasons
    annotations: Dict[str, List[Tuple[str, List[float]]]] = defaultdict(list)
    invalid_lines: List[Dict[str, Any]] = []
    with zipfile.ZipFile(export_path) as archive:
        # Class names from obj.names, normalised to canonical labels; a line's class ID indexes this list
        names_content = archive.read("obj.names").decode("utf-8-sig")
        labels = [normalize_label(line) for line in names_content.splitlines() if line.strip()]

        # One YOLO label file per image under obj_train_data/
        members = sorted(
            n for n in archive.namelist() if n.startswith("obj_train_data/") and n.endswith(".txt")
        )
        for member in members:
            stem = Path(member).stem
            content = archive.read(member).decode("utf-8-sig")
            for line_number, line in enumerate(content.splitlines(), start=1):
                # Each line is "class cx cy w h" in normalised coordinates; blank lines are skipped
                parts = line.strip().split()
                if not parts:
                    continue

                # Record a rejected line with its export, image stem, line number and reason
                def reject(reason: str) -> None:
                    invalid_lines.append({
                        "annotation_export": export_path.name,
                        "stem": stem,
                        "line_number": line_number,
                        "line": line.strip(),
                        "reason": reason,
                    })

                # Reject lines with the wrong field count, an unknown class or a box outside the image
                if len(parts) != 5:
                    reject("wrong_field_count")
                    continue
                class_id = int(float(parts[0]))
                if class_id < 0 or class_id >= len(labels):
                    reject("class_id_out_of_range")
                    continue
                box = [float(v) for v in parts[1:]]
                if any(v < 0 or v > 1 for v in box) or box[2] <= 0 or box[3] <= 0:
                    reject("box_out_of_bounds")
                    continue
                # Keep the valid box under its image stem with its class name
                annotations[stem].append((labels[class_id], box))

    return labels, annotations, invalid_lines


def parse_cvat_xml_export(
    export_path: Path,
) -> Tuple[List[str], Dict[str, List[Tuple[str, List[float]]]], List[Dict[str, Any]]]:
    """Parse boxes from a CVAT for images 1.1 export (``annotations.xml``), as ``parse_yolo_export`` does.

    Pixel boxes are converted to the YOLO frame (centre and size, 0-1) with the image size the XML
    records. A rotated box becomes the axis-aligned rectangle that encloses it, so the crop holds the
    whole marked wound (YOLO 1.1 cannot store a rotation). Shapes other than boxes are rejected.
    """
    annotations: Dict[str, List[Tuple[str, List[float]]]] = defaultdict(list)
    invalid_lines: List[Dict[str, Any]] = []
    with zipfile.ZipFile(export_path) as archive:
        root = ET.fromstring(archive.read("annotations.xml"))
    # Class names from the label list, normalised to canonical labels
    labels = [normalize_label(node.text or "") for node in root.iterfind("./meta/*/labels/label/name")]

    for image in root.iter("image"):
        stem = Path(image.get("name", "")).stem
        width, height = float(image.get("width", 0)), float(image.get("height", 0))
        for shape in image:
            # Record a rejected shape with its export, image stem, shape number and reason
            def reject(reason: str) -> None:
                invalid_lines.append({
                    "annotation_export": export_path.name,
                    "stem": stem,
                    "line_number": int(image.get("id", -1)),
                    "line": ET.tostring(shape, encoding="unicode").strip()[:200],
                    "reason": reason,
                })

            if shape.tag != "box":
                reject("not_a_box")
                continue
            label = normalize_label(shape.get("label", ""))
            if label not in labels:
                reject("class_id_out_of_range")
                continue
            x1, y1, x2, y2 = (float(shape.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            # Enclosing axis-aligned size of a box rotated about its centre
            angle = math.radians(float(shape.get("rotation", 0)))
            bw = (x2 - x1) * abs(math.cos(angle)) + (y2 - y1) * abs(math.sin(angle))
            bh = (x2 - x1) * abs(math.sin(angle)) + (y2 - y1) * abs(math.cos(angle))
            box = [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, bw / width, bh / height] if width and height else []
            if not box or any(v < 0 or v > 1 for v in box) or box[2] <= 0 or box[3] <= 0:
                reject("box_out_of_bounds")
                continue
            annotations[stem].append((label, box))

    return labels, annotations, invalid_lines


def parse_export(
    export_path: Path,
) -> Tuple[List[str], Dict[str, List[Tuple[str, List[float]]]], List[Dict[str, Any]]]:
    """Parse a CVAT export zip: CVAT for images 1.1 if it holds ``annotations.xml``, else YOLO 1.1."""
    with zipfile.ZipFile(export_path) as archive:
        is_xml = "annotations.xml" in archive.namelist()
    return parse_cvat_xml_export(export_path) if is_xml else parse_yolo_export(export_path)


def compute_union_box(boxes: List[List[float]], width: int, height: int, padding: float) -> Tuple[int, int, int, int]:
    """Compute padded union bounding box in pixel coordinates."""
    # Convert the normalised centre boxes to pixels and take their enclosing rectangle
    x1 = min((cx - bw / 2.0) * width for cx, cy, bw, bh in boxes)
    y1 = min((cy - bh / 2.0) * height for cx, cy, bw, bh in boxes)
    x2 = max((cx + bw / 2.0) * width for cx, cy, bw, bh in boxes)
    y2 = max((cy + bh / 2.0) * height for cx, cy, bw, bh in boxes)

    # Pad each side by a share of the box size, clipped to the image borders
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding

    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )


def find_source_image(image_root: Path, row: Dict[str, Any]) -> Optional[Path]:
    """Locate a manifest row's raw image under ``image_root``.

    The raw tree is organised by raw ``wound_label`` folder (e.g. ``gsw/``), and some manifest
    filenames omit the extension (``IMG_0235`` for ``IMG_0235.JPG``).
    """
    # Try the folder the photo was filed under before relabelling (T23 source_label), the raw label
    # folder, then the canonical label folder, then the root
    fname = str(row["filename"]).strip()
    folders = [str(row.get(c, "")).strip() for c in ("source_label", "wound_label", "canonical_wound_label")]
    for folder in dict.fromkeys([f for f in folders if f] + [""]):
        directory = image_root / folder if folder else image_root
        # A filename with an image extension must exist exactly as written
        if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
            candidate = directory / fname
            if candidate.is_file():
                return candidate
            continue
        # Without an extension, accept a single image file with that stem; several are ambiguous
        matches = sorted(
            p for p in directory.glob(f"{glob.escape(fname)}.*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(matches) > 1:
            raise ValueError(f"Ambiguous source image for {fname!r}: {[str(m) for m in matches]}")
        if matches:
            return matches[0]
    return None


def load_dimension_cache(reference_manifest: Optional[Path]) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Load source image dimensions keyed by (canonical label, stem) from a previous ROI manifest.

    Dry runs use this when the raw images live on the HPC cluster rather than on local disk.
    """
    # No reference manifest means an empty cache
    cache: Dict[Tuple[str, str], Tuple[int, int]] = {}
    if reference_manifest is None or not reference_manifest.exists():
        return cache
    # Read the reference and key each image's width and height by (label, stem)
    df = pd.read_csv(reference_manifest, dtype=str)
    missing = {"filename", "source_width", "source_height"}.difference(df.columns)
    if missing:
        raise ValueError(f"Reference manifest {reference_manifest} missing columns: {sorted(missing)}")
    for row in df.to_dict("records"):
        cache[manifest_key(row)] = (int(row["source_width"]), int(row["source_height"]))
    return cache


def resolve_export_files(exports_arg: List[Path]) -> List[Path]:
    """Expand export paths (zip files or directories of zips), skipping byte-identical copies.

    Shell globs such as ``task_*_annotations_*.zip`` also match browser re-downloads like
    ``task_17_... (1).zip``; parsing the same archive twice would collide on every annotation.
    """
    # Collect the zip files given directly or found in the given directories
    candidates: List[Path] = []
    for item in exports_arg:
        item = Path(item).expanduser()
        if item.is_dir():
            candidates.extend(sorted(item.glob("*.zip")))
        elif item.is_file():
            candidates.append(item)
        else:
            raise FileNotFoundError(f"Annotation export not found: {item}")

    # Keep one file per content hash, preferring the name without a " (n)" suffix
    kept_by_digest: Dict[str, Path] = {}
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        kept = kept_by_digest.get(digest)
        if kept is None or (BROWSER_COPY_SUFFIX.search(kept.stem) and not BROWSER_COPY_SUFFIX.search(path.stem)):
            kept_by_digest[digest] = path
    # Return the kept files in their original order
    kept_paths = set(kept_by_digest.values())
    return [p for p in dict.fromkeys(candidates) if p in kept_paths]


@dataclass
class RoiBuildResult:
    rows: List[Dict[str, Any]] = field(default_factory=list)
    status: Counter = field(default_factory=Counter)
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_annotations: List[Dict[str, str]] = field(default_factory=list)
    invalid_annotation_lines: List[Dict[str, Any]] = field(default_factory=list)


def build_roi_records(
    source_rows: List[Dict[str, Any]],
    exports: List[Path],
    *,
    dimensions: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None,
    padding: float = 0.08,
    min_size: int = 8,
    crop_mode: str = "union",
    image_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    group_col: str = "pseudonimized_id",
) -> RoiBuildResult:
    """Match CVAT annotations to manifest rows and compute one padded crop box per source image.

    Every ROI row copies its source manifest row, so ``pseudonimized_id`` and ``split`` are
    inherited unchanged. Manifest rows that yield no crop are reported in ``dropped`` with a reason.

    Dry run when ``image_root`` is None: dimensions come from ``dimensions`` and nothing is written.
    Live mode: dimensions come from the EXIF-oriented source image and crops are saved under
    ``output_dir/<split>/<label>/``.
    """
    # Index the manifest rows by (canonical label, filename stem), the key annotations use
    lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        lookup[manifest_key(row)].append(row)
    dimensions = dimensions or {}

    # Results, and which export each (label, stem) was annotated in, to catch collisions
    result = RoiBuildResult()
    annotated_in: Dict[Tuple[str, str], str] = {}

    # Record a manifest row that yields no crop, with the reason
    def drop(row: Dict[str, Any], reason: str) -> None:
        result.dropped.append({
            group_col: row[group_col],
            "filename": row["filename"],
            "canonical_wound_label": class_label(row),
            "split": row["split"],
            "reason": reason,
        })
        result.status[f"dropped_{reason}"] += 1

    for export in exports:
        # Parse each export and keep its rejected lines for the summary
        _, annotations, invalid_lines = parse_export(export)
        result.invalid_annotation_lines.extend(invalid_lines)
        result.status["invalid_annotation_lines"] += len(invalid_lines)
        for stem, labeled_boxes in sorted(annotations.items()):
            # Group this image's boxes by wound class
            boxes_by_label: Dict[str, List[List[float]]] = defaultdict(list)
            for label, box in labeled_boxes:
                boxes_by_label[label].append(box)

            for label, boxes in boxes_by_label.items():
                # The same (label, stem) annotated in two exports is an error
                key = (label, stem)
                if key in annotated_in:
                    raise ValueError(f"Annotation collision for {key}: {annotated_in[key]} and {export.name}")
                annotated_in[key] = export.name

                # Find the single manifest row for this annotation; unmatched ones are only counted
                candidates = lookup.get(key, [])
                if not candidates:
                    result.unmatched_annotations.append(
                        {"stem": stem, "label": label, "annotation_export": export.name}
                    )
                    result.status["annotation_not_in_manifest"] += 1
                    continue
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous manifest match for annotation {key}: {len(candidates)} rows")
                source_row = candidates[0]

                # Image size: from the EXIF-rotated source image in live mode, else from the dimension cache
                image: Optional[Image.Image] = None
                source_relative_path = ""
                if image_root is not None:
                    source_path = find_source_image(image_root, source_row)
                    if source_path is None:
                        drop(source_row, "source_image_missing")
                        continue
                    with Image.open(source_path) as opened:
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                    source_w, source_h = image.size
                    source_relative_path = source_path.relative_to(image_root).as_posix()
                elif key in dimensions:
                    source_w, source_h = dimensions[key]
                else:
                    drop(source_row, "missing_dimensions")
                    continue

                # Split name and training label for the crop path (the label differs from the
                # annotation's only for a photo relabelled after annotation); union mode makes one crop
                # from all boxes, individual mode one per box
                split = source_row["split"].strip().lower()
                crop_label = class_label(source_row)
                crop_groups = [boxes] if crop_mode == "union" else [[b] for b in boxes]
                for crop_index, crop_boxes in enumerate(crop_groups, start=1):
                    # Padded crop box in pixels; drop crops narrower or shorter than the minimum size
                    x1, y1, x2, y2 = compute_union_box(crop_boxes, source_w, source_h, padding)
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        drop(source_row, "small_crop")
                        continue

                    # Crop path is <split>/<label>/<stem>_<suffix>.jpg; save the JPEG only in live mode
                    suffix = "roi" if crop_mode == "union" else f"bbox{crop_index:02d}"
                    roi_relative_path = f"{split}/{safe_folder(crop_label)}/{stem}_{suffix}.jpg"
                    if image is not None and output_dir is not None:
                        out_path = output_dir / roi_relative_path
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        image.crop((x1, y1, x2, y2)).save(out_path, format="JPEG", quality=95)

                    # Copy the source manifest row (case and split inherited) and add the crop details
                    out_row = dict(source_row)
                    out_row.update({
                        "canonical_wound_label": crop_label,
                        "source_relative_path": source_relative_path,
                        "roi_relative_path": roi_relative_path,
                        "annotation_export": export.name,
                        "crop_mode": crop_mode,
                        "crop_index": crop_index,
                        "bbox_count": len(crop_boxes),
                        "padding_ratio": padding,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "source_width": source_w,
                        "source_height": source_h,
                    })
                    result.rows.append(out_row)
                    result.status["crops_created"] += 1

    # Manifest rows that no export annotated are dropped as no_annotation
    for row in source_rows:
        if manifest_key(row) not in annotated_in:
            drop(row, "no_annotation")

    # Final counts of manifest rows and rows without a usable annotation
    result.status["manifest_rows"] = len(source_rows)
    result.status["manifest_rows_without_usable_annotation"] = result.status["dropped_no_annotation"]
    return result


def main() -> int:
    args = parse_args()
    # Resolve paths and check the mode: a live run needs an existing image root
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    group_col = args.group_column

    if not args.dry_run and args.image_root is None:
        print("Error: live mode requires --image-root (or pass --dry-run).", file=sys.stderr)
        return 1
    if args.image_root is not None and not args.dry_run and not args.image_root.is_dir():
        print(f"Error: image root not found: {args.image_root}", file=sys.stderr)
        return 1

    # Read and validate the case-level split manifest
    print(f"Reading authoritative split manifest: {manifest_path}")
    source_rows = read_manifest(manifest_path, group_col)

    # Resolve the export archives, dropping byte-identical duplicates, and list what is used
    candidate_exports = [Path(p).expanduser() for p in args.exports]
    export_files = resolve_export_files(candidate_exports)
    if not export_files:
        print("Error: No CVAT annotation export archives found.", file=sys.stderr)
        return 1
    duplicate_exports = sorted(
        p.name for p in candidate_exports if p.is_file() and p not in export_files
    )

    print(f"Using {len(export_files)} CVAT export archives:")
    for exp in export_files:
        print(f"  - {exp.name}")
    for name in duplicate_exports:
        print(f"  (skipped byte-identical duplicate: {name})")

    # A dry run takes image sizes from the reference manifest instead of reading images
    dimension_cache: Dict[Tuple[str, str], Tuple[int, int]] = {}
    if args.dry_run:
        if not args.reference_manifest.exists():
            print(f"Error: dry run needs image dimensions from --reference-manifest; not found: "
                  f"{args.reference_manifest}", file=sys.stderr)
            return 1
        dimension_cache = load_dimension_cache(args.reference_manifest)
        print(f"Loaded image dimension cache for {len(dimension_cache)} images from {args.reference_manifest}")

    # Optionally wipe the previous output (live mode only), then make sure the folder exists
    if args.clear_output and output_dir.exists() and not args.dry_run:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match annotations to manifest rows and compute (and in live mode save) the crops
    result = build_roi_records(
        source_rows,
        export_files,
        dimensions=dimension_cache,
        padding=args.padding,
        min_size=args.min_size,
        crop_mode=args.crop_mode,
        image_root=None if args.dry_run else args.image_root.resolve(),
        output_dir=None if args.dry_run else output_dir,
        group_col=group_col,
    )
    # The ROI rows must still keep every case in a single split
    verify_no_case_leakage(result.rows, group_col)

    # Write the ROI manifest, one row per crop
    manifest_out = output_dir / "roi_manifest.csv"
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.rows[0].keys()) if result.rows else [])
        writer.writeheader()
        writer.writerows(result.rows)

    # Crop and case counts per split and class, and the dropped rows split by reason type
    crops_by_split = dict(Counter(r["split"] for r in result.rows))
    crops_by_class = dict(Counter(r["canonical_wound_label"] for r in result.rows))
    cases_by_split = {
        split: len({r[group_col] for r in result.rows if r["split"] == split}) for split in SPLIT_NAMES
    }
    crops_by_split_and_class = {
        split: dict(sorted(Counter(r["canonical_wound_label"] for r in result.rows if r["split"] == split).items()))
        for split in SPLIT_NAMES
    }
    unannotated = [d for d in result.dropped if d["reason"] == "no_annotation"]
    skipped = [d for d in result.dropped if d["reason"] != "no_annotation"]

    # Settings, counts and every dropped or rejected record, written to crop_summary.json
    summary = {
        "source_manifest": str(manifest_path),
        "image_root": None if args.dry_run else str(args.image_root.resolve()),
        "output_dir": str(output_dir),
        "annotation_exports": [p.name for p in export_files],
        "duplicate_exports_skipped": duplicate_exports,
        "crop_mode": args.crop_mode,
        "padding_ratio": args.padding,
        "min_size": args.min_size,
        "dry_run": args.dry_run,
        "counts": dict(sorted(result.status.items())),
        "unique_cases": len({r[group_col] for r in result.rows}),
        "cases_by_split": cases_by_split,
        "crops_by_split": crops_by_split,
        "crops_by_class": crops_by_class,
        "crops_by_split_and_class": crops_by_split_and_class,
        "unannotated_records_count": len(unannotated),
        "unannotated_records": unannotated,
        "skipped_records": skipped,
        "annotations_not_in_manifest": result.unmatched_annotations,
        "invalid_annotation_lines": result.invalid_annotation_lines,
    }

    summary_out = output_dir / "crop_summary.json"
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print the totals and the crops per split and class
    print("\n--- ROI Dataset Preprocessing Summary ---")
    print(f"Total Source Manifest Rows:  {result.status['manifest_rows']:,}")
    print(f"Total ROI Crops Generated:   {result.status['crops_created']:,} across {summary['unique_cases']} cases")
    print(f"Unannotated Source Images:   {len(unannotated)}")
    print(f"Annotations Not In Manifest: {len(result.unmatched_annotations)}")
    print(f"Invalid Annotation Lines:    {len(result.invalid_annotation_lines)}")
    print(f"Skipped Annotated Records:   {len(skipped)} (too small / missing image / missing dimensions)")
    print("\nCrops by Split:")
    for s_name in SPLIT_NAMES:
        print(f"  - {s_name:<12}: {crops_by_split.get(s_name, 0):>4} images, {cases_by_split[s_name]:>3} cases")
    print("\nCrops by Canonical Class:")
    for c_name, c_cnt in sorted(crops_by_class.items()):
        print(f"  - {c_name:<20}: {c_cnt:>4}")

    print(f"\nManifest saved: {manifest_out}")
    print(f"Summary saved:  {summary_out}")

    # Warn when annotated images were skipped, since those are real losses
    if skipped:
        print(f"\nWarning: {len(skipped)} annotated records were skipped; see skipped_records in {summary_out}",
              file=sys.stderr)
    print("\nProcessing completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
