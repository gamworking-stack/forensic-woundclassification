"""Synthetic wound dataset for training tests: raw tree, ROI crops and both manifests."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


RAW_FOLDERS = {
    "contusion": "contusion",
    "cut wound": "cut",
    "graze wound": "graze",
    "gunshot wound": "gsw",
    "laceration": "laceration",
    "penetration wound": "penetrating wound (sharp&blunt)",
    "scratch": "scratch",
}
SPLITS = ("train", "validation", "test")
IMAGES_PER_CLASS = 2
EXPECTED_PREDICTION_COLUMNS = [
    "relative_path", "pseudonimized_id", "split", "true_class", "predicted_class",
    *[f"prob_{c.replace(' ', '_')}" for c in sorted(RAW_FOLDERS)],
]


def build_synthetic_dataset(root: Path) -> dict:
    """Raw tree by raw label, ROI crops by split/class, and both manifests; one image per case.

    Every train class also gets one extra full image with no ROI crop, so 'matched' and 'all'
    scopes differ. One filename is listed without its extension, as in the real manifest.
    """
    rng = np.random.default_rng(0)
    full_rows, roi_rows = [], []
    case_id = 0
    for split in SPLITS:
        for label, folder in RAW_FOLDERS.items():
            extra = 1 if split == "train" else 0
            for k in range(IMAGES_PER_CLASS + extra):
                case_id += 1
                stem = f"IMG_{case_id:04d}"
                raw_dir = root / "raw" / folder
                raw_dir.mkdir(parents=True, exist_ok=True)
                pixels = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(raw_dir / f"{stem}.JPG")
                filename = stem if case_id == 1 else f"{stem}.JPG"
                row = {
                    "pseudonimized_id": str(case_id),
                    "wound_id": "1.0",
                    "filename": filename,
                    "wound_label": folder,
                    "canonical_wound_label": label,
                    "relative_path": f"{label}/{filename}",
                    "split": split,
                }
                full_rows.append(row)
                if k < IMAGES_PER_CLASS:
                    roi_rel = f"{split}/{label}/{stem}_roi.jpg"
                    (root / "roi" / split / label).mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pixels[10:50, 10:70]).save(root / "roi" / roi_rel)
                    roi_rows.append({**row, "roi_relative_path": roi_rel})

    full_manifest = root / "full_manifest.csv"
    roi_manifest = root / "roi_manifest.csv"
    pd.DataFrame(full_rows).to_csv(full_manifest, index=False)
    pd.DataFrame(roi_rows).to_csv(roi_manifest, index=False)
    return {
        "full_manifest": full_manifest,
        "roi_manifest": roi_manifest,
        "raw_root": root / "raw",
        "roi_root": root / "roi",
    }
