#!/usr/bin/env python3
"""Create a leakage-safe ROI classification dataset from CVAT YOLO exports.

Parses CVAT YOLO 1.1 annotation archives, matches annotations with the authoritative
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
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from PIL import Image, ImageFile, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.validate_metadata import LABEL_NORMALIZATION_MAP  # noqa: E402


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
        default=Path("data/splits/case_level_split_manifest.csv"),
        help="Path to authoritative case-level split manifest.",
    )
    parser.add_argument(
        "--exports",
        type=Path,
        nargs="+",
        required=True,
        help="CVAT YOLO 1.1 annotation .zip files or directory containing them.",
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
        default=Path("data/roi"),
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

    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {group_col, "split", "filename"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(f"Manifest missing required columns: {required.difference(rows[0].keys() if rows else set())}")

    for row in rows:
        if row["split"].strip().lower() not in SPLIT_NAMES:
            raise ValueError(f"Unexpected split value: {row['split']!r}")

    verify_no_case_leakage(rows, group_col)
    return rows


def verify_no_case_leakage(rows: List[Dict[str, Any]], group_col: str) -> None:
    """Raise if any case appears in more than one split."""
    groups_by_split: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        groups_by_split[str(row["split"]).strip().lower()].add(str(row[group_col]))
    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1:]:
            shared = groups_by_split[first] & groups_by_split[second]
            if shared:
                raise AssertionError(f"Case leakage between {first} and {second}: {sorted(shared)}")


def manifest_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """Join key between manifest rows and annotations: (canonical label, filename stem)."""
    label = normalize_label(row.get("canonical_wound_label") or row.get("wound_label", ""))
    return label, Path(str(row["filename"]).strip()).stem


def parse_yolo_export(
    export_path: Path,
) -> Tuple[List[str], Dict[str, List[Tuple[str, List[float]]]], List[Dict[str, Any]]]:
    """Parse YOLO annotations from a CVAT zip export.

    Returns the normalized class names, the valid boxes per image stem, and every rejected line
    with the reason it was rejected.
    """
    annotations: Dict[str, List[Tuple[str, List[float]]]] = defaultdict(list)
    invalid_lines: List[Dict[str, Any]] = []
    with zipfile.ZipFile(export_path) as archive:
        names_content = archive.read("obj.names").decode("utf-8-sig")
        labels = [normalize_label(line) for line in names_content.splitlines() if line.strip()]

        members = sorted(
            n for n in archive.namelist() if n.startswith("obj_train_data/") and n.endswith(".txt")
        )
        for member in members:
            stem = Path(member).stem
            content = archive.read(member).decode("utf-8-sig")
            for line_number, line in enumerate(content.splitlines(), start=1):
                parts = line.strip().split()
                if not parts:
                    continue

                def reject(reason: str) -> None:
                    invalid_lines.append({
                        "annotation_export": export_path.name,
                        "stem": stem,
                        "line_number": line_number,
                        "line": line.strip(),
                        "reason": reason,
                    })

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
                annotations[stem].append((labels[class_id], box))

    return labels, annotations, invalid_lines


def compute_union_box(boxes: List[List[float]], width: int, height: int, padding: float) -> Tuple[int, int, int, int]:
    """Compute padded union bounding box in pixel coordinates."""
    x1 = min((cx - bw / 2.0) * width for cx, cy, bw, bh in boxes)
    y1 = min((cy - bh / 2.0) * height for cx, cy, bw, bh in boxes)
    x2 = max((cx + bw / 2.0) * width for cx, cy, bw, bh in boxes)
    y2 = max((cy + bh / 2.0) * height for cx, cy, bw, bh in boxes)

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
    fname = str(row["filename"]).strip()
    folders = [str(row.get("wound_label", "")).strip(), str(row.get("canonical_wound_label", "")).strip(), ""]
    for folder in dict.fromkeys(folders):
        directory = image_root / folder if folder else image_root
        if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
            candidate = directory / fname
            if candidate.is_file():
                return candidate
            continue
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
    cache: Dict[Tuple[str, str], Tuple[int, int]] = {}
    if reference_manifest is None or not reference_manifest.exists():
        return cache
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
    candidates: List[Path] = []
    for item in exports_arg:
        item = Path(item).expanduser()
        if item.is_dir():
            candidates.extend(sorted(item.glob("*.zip")))
        elif item.is_file():
            candidates.append(item)
        else:
            raise FileNotFoundError(f"Annotation export not found: {item}")

    kept_by_digest: Dict[str, Path] = {}
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        kept = kept_by_digest.get(digest)
        if kept is None or (BROWSER_COPY_SUFFIX.search(kept.stem) and not BROWSER_COPY_SUFFIX.search(path.stem)):
            kept_by_digest[digest] = path
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
    lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        lookup[manifest_key(row)].append(row)
    dimensions = dimensions or {}

    result = RoiBuildResult()
    annotated_in: Dict[Tuple[str, str], str] = {}

    def drop(row: Dict[str, Any], reason: str) -> None:
        result.dropped.append({
            group_col: row[group_col],
            "filename": row["filename"],
            "canonical_wound_label": manifest_key(row)[0],
            "split": row["split"],
            "reason": reason,
        })
        result.status[f"dropped_{reason}"] += 1

    for export in exports:
        _, annotations, invalid_lines = parse_yolo_export(export)
        result.invalid_annotation_lines.extend(invalid_lines)
        result.status["invalid_annotation_lines"] += len(invalid_lines)
        for stem, labeled_boxes in sorted(annotations.items()):
            boxes_by_label: Dict[str, List[List[float]]] = defaultdict(list)
            for label, box in labeled_boxes:
                boxes_by_label[label].append(box)

            for label, boxes in boxes_by_label.items():
                key = (label, stem)
                if key in annotated_in:
                    raise ValueError(f"Annotation collision for {key}: {annotated_in[key]} and {export.name}")
                annotated_in[key] = export.name

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

                split = source_row["split"].strip().lower()
                crop_groups = [boxes] if crop_mode == "union" else [[b] for b in boxes]
                for crop_index, crop_boxes in enumerate(crop_groups, start=1):
                    x1, y1, x2, y2 = compute_union_box(crop_boxes, source_w, source_h, padding)
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        drop(source_row, "small_crop")
                        continue

                    suffix = "roi" if crop_mode == "union" else f"bbox{crop_index:02d}"
                    roi_relative_path = f"{split}/{safe_folder(label)}/{stem}_{suffix}.jpg"
                    if image is not None and output_dir is not None:
                        out_path = output_dir / roi_relative_path
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        image.crop((x1, y1, x2, y2)).save(out_path, format="JPEG", quality=95)

                    out_row = dict(source_row)
                    out_row.update({
                        "canonical_wound_label": label,
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

    for row in source_rows:
        if manifest_key(row) not in annotated_in:
            drop(row, "no_annotation")

    result.status["manifest_rows"] = len(source_rows)
    result.status["manifest_rows_without_usable_annotation"] = result.status["dropped_no_annotation"]
    return result


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    group_col = args.group_column

    if not args.dry_run and args.image_root is None:
        print("Error: live mode requires --image-root (or pass --dry-run).", file=sys.stderr)
        return 1
    if args.image_root is not None and not args.dry_run and not args.image_root.is_dir():
        print(f"Error: image root not found: {args.image_root}", file=sys.stderr)
        return 1

    print(f"Reading authoritative split manifest: {manifest_path}")
    source_rows = read_manifest(manifest_path, group_col)

    candidate_exports = [Path(p).expanduser() for p in args.exports]
    export_files = resolve_export_files(candidate_exports)
    if not export_files:
        print("Error: No CVAT annotation export archives found.", file=sys.stderr)
        return 1
    duplicate_exports = sorted(
        p.name for p in candidate_exports if p.is_file() and p not in export_files
    )

    print(f"Using {len(export_files)} CVAT YOLO export archives:")
    for exp in export_files:
        print(f"  - {exp.name}")
    for name in duplicate_exports:
        print(f"  (skipped byte-identical duplicate: {name})")

    dimension_cache: Dict[Tuple[str, str], Tuple[int, int]] = {}
    if args.dry_run:
        if not args.reference_manifest.exists():
            print(f"Error: dry run needs image dimensions from --reference-manifest; not found: "
                  f"{args.reference_manifest}", file=sys.stderr)
            return 1
        dimension_cache = load_dimension_cache(args.reference_manifest)
        print(f"Loaded image dimension cache for {len(dimension_cache)} images from {args.reference_manifest}")

    if args.clear_output and output_dir.exists() and not args.dry_run:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    verify_no_case_leakage(result.rows, group_col)

    manifest_out = output_dir / "roi_manifest.csv"
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.rows[0].keys()) if result.rows else [])
        writer.writeheader()
        writer.writerows(result.rows)

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

    if skipped:
        print(f"\nWarning: {len(skipped)} annotated records were skipped; see skipped_records in {summary_out}",
              file=sys.stderr)
    print("\nProcessing completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
