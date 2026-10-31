#!/usr/bin/env python3
"""T25: copies of finished runs' prediction files with one case per patient.

Every ``*_predictions.csv`` under ``--source-root`` is copied to the same place under ``--output-root``
with ``pseudonimized_id`` (the T23 group) replaced by the photo's ``patient_case``
(``src/data/patient_cases.py``). Nothing else changes, so evaluate.py, summarize.py, crop_summary.py
and subgroup.py run on the copies unchanged and their case-level bootstrap resamples patients.
A ``split_manifest.csv`` with the same replacement is written beside them for subgroup.py.

Rows are matched to their photo as subgroup.py matches them: crops and photo-level rows by path,
full images by group and file stem (the file names the raw photo). Every row must match a photo of
its own group. Outputs go outside the repository, like every per-image file of the T23 runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.data.patient_cases import PATIENT_CASE  # noqa: E402
from src.evaluation.metrics import ID_COLUMNS  # noqa: E402
from src.evaluation.single_box import photo_stem  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy prediction files with one case per patient.")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Folder holding copied runs (e.g. main/, roi_box/, roi_crop/).")
    parser.add_argument("--output-root", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--patient-cases", type=Path, default=DATA_DIR / "splits/patient_cases.csv")
    parser.add_argument("--split-manifest", type=Path, default=DATA_DIR / "splits/case_level_split_manifest.csv")
    parser.add_argument("--roi-manifest", type=Path, default=DATA_DIR / "roi/roi_manifest.csv")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    return parser.parse_args(argv)


def lookups(patients: pd.DataFrame, roi_manifest: Path, box_manifest: Path
            ) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
    """Path of every photo and crop -> photo; (group, file stem) -> photo for full images."""
    by_path = dict(zip(patients["relative_path"], patients["relative_path"]))
    for manifest in (roi_manifest, box_manifest):
        crops = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        by_path.update(zip(crops["roi_relative_path"], crops["relative_path"]))
    by_stem: Dict[Tuple[str, str], str] = {}
    for group, name, photo in zip(patients["pseudonimized_id"], patients["filename"], patients["relative_path"]):
        key = (group, photo_stem(name))
        if key in by_stem:
            raise ValueError(f"Group {group} has two photos named {key[1]}; full-image rows are ambiguous")
        by_stem[key] = photo
    return by_path, by_stem


def remap(frame: pd.DataFrame, patients: pd.DataFrame, by_path: Dict, by_stem: Dict, source: Path) -> pd.DataFrame:
    """The prediction rows with ``pseudonimized_id`` set to their photo's patient case."""
    photo = frame["relative_path"].map(by_path)
    stems = pd.Series(list(zip(frame["pseudonimized_id"], frame["relative_path"].map(photo_stem))), index=frame.index)
    photo = photo.fillna(stems.map(by_stem))
    if photo.isna().any():
        raise ValueError(f"{source}: {int(photo.isna().sum())} rows match no photo "
                         f"(e.g. {frame.loc[photo.isna(), 'relative_path'].head(3).tolist()})")
    info = patients.set_index("relative_path").loc[photo]
    if (info["pseudonimized_id"].to_numpy() != frame["pseudonimized_id"].to_numpy()).any():
        raise ValueError(f"{source}: a row matched a photo of another group")
    out = frame.copy()
    out["pseudonimized_id"] = info[PATIENT_CASE].to_numpy()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_root = args.output_root.resolve()
    if out_root == REPO_ROOT or REPO_ROOT in out_root.parents:
        raise ValueError(f"--output-root {out_root} is inside the repository; per-image files stay outside it")
    patients = pd.read_csv(args.patient_cases, dtype=str, keep_default_na=False)
    by_path, by_stem = lookups(patients, args.roi_manifest, args.box_manifest)

    files = sorted(args.source_root.rglob("*_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No *_predictions.csv under {args.source_root}")
    for path in files:
        frame = pd.read_csv(path, dtype={c: str for c in ID_COLUMNS})
        target = out_root / path.relative_to(args.source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        remap(frame, patients, by_path, by_stem, path).to_csv(target, index=False)

    # The split manifest with the same cases, for subgroup.py --split-manifest
    manifest = pd.read_csv(args.split_manifest, dtype=str, keep_default_na=False)
    case = manifest["relative_path"].map(dict(zip(patients["relative_path"], patients[PATIENT_CASE])))
    if case.isna().any():
        raise ValueError(f"{args.split_manifest}: {int(case.isna().sum())} photos have no patient case")
    manifest.assign(pseudonimized_id=case).to_csv(out_root / "split_manifest.csv", index=False)
    print(f"{len(files)} prediction files rewritten with {patients[PATIENT_CASE].nunique()} patient cases "
          f"to {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
