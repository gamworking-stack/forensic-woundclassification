#!/usr/bin/env python3
"""T30: a contrast-enhanced copy of the raw photographs (CLAHE), for the "more contrast" experiment.

Every image under ``--image-root`` is written to the same relative path under ``--output-dir`` after
Contrast Limited Adaptive Histogram Equalization on the lightness channel only (CIELAB L*; a* and b*
untouched, so wound colour is kept). The EXIF orientation is applied first and the copy carries no
EXIF, so every reader (training, the crop job) sees the same upright image at the same size as the
original. Crops are then cut from these copies by the existing crop job with the same boxes, and the
models are retrained on them; nothing else in the pipeline changes.

Writes ``contrast_manifest.csv`` (relative path, width, height) and ``contrast_report.json``
(parameters, counts) beside the images. Patient images: run on Raptor, outputs stay there.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_roi_dataset import IMAGE_EXTENSIONS  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True
CLIP_LIMIT = 2.0
TILE_GRID = 8
JPEG_QUALITY = 95


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLAHE copy of an image tree.")
    parser.add_argument("--image-root", type=Path, required=True, help="Raw photographs (read only).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Mirror of --image-root, enhanced.")
    parser.add_argument("--clip-limit", type=float, default=CLIP_LIMIT)
    parser.add_argument("--tile-grid", type=int, default=TILE_GRID, help="Tiles per side.")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def clahe_rgb(image: Image.Image, clip_limit: float = CLIP_LIMIT, tile_grid: int = TILE_GRID) -> Image.Image:
    """CLAHE on L* of an RGB image; a* and b* (colour) are left as they are."""
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def enhance_file(source: Path, target: Path, clip_limit: float, tile_grid: int, quality: int) -> Dict:
    # Upright first (EXIF orientation), then enhance; the copy has no EXIF, so it is read upright as is
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    enhanced = clahe_rgb(image, clip_limit, tile_grid)
    target.parent.mkdir(parents=True, exist_ok=True)
    enhanced.save(target, format="JPEG", quality=quality, subsampling=0)
    return {"width": enhanced.width, "height": enhanced.height}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root, out = args.image_root.resolve(), args.output_dir.resolve()
    if out == root or root in out.parents:
        raise ValueError(f"--output-dir {out} must not be the image root or inside it")
    # Hidden files are skipped: macOS copies leave "._<name>.JPG" metadata files beside the photos
    sources = sorted(p for p in root.rglob("*")
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))
    if not sources:
        raise FileNotFoundError(f"No images under {root}")
    relative = [p.relative_to(root) for p in sources]

    # One process per image: decoding and CLAHE on 24-megapixel photos is CPU bound
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        sizes = list(pool.map(enhance_file, sources, [out / r for r in relative],
                              [args.clip_limit] * len(sources), [args.tile_grid] * len(sources),
                              [args.quality] * len(sources), chunksize=4))

    manifest = pd.DataFrame({"relative_path": [r.as_posix() for r in relative],
                             "width": [s["width"] for s in sizes], "height": [s["height"] for s in sizes]})
    manifest.to_csv(out / "contrast_manifest.csv", index=False)
    report = {"method": "CLAHE on CIELAB L* (OpenCV)", "clip_limit": args.clip_limit,
              "tile_grid": [args.tile_grid, args.tile_grid], "jpeg_quality": args.quality,
              "exif_orientation_applied": True, "images": len(sources), "opencv": cv2.__version__,
              "image_root": str(root)}
    (out / "contrast_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(sources)} images enhanced (CLAHE clip {args.clip_limit}, {args.tile_grid}x{args.tile_grid} tiles) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
