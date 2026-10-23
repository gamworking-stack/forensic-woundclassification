#!/usr/bin/env python3
"""Union and per-box crops from one detector arm's auto boxes (T19 §2.3 step 3, §3.5).

Input: the arm's frozen ``selection.json`` and the detections files of its chosen configuration
covering every photo once:
- 4Z and 4P: the development file plus the test file;
- 4F: the five cross-fitted ``train_fold<k>.csv``, the validation file and the test file.

A photo in two files is an error.

Every photo of the per-box manifest gets its final boxes (``candidate_boxes`` then ``final_boxes``
with the arm's threshold and the training-split cap and oversize share), and from them:
- ``union/roi_manifest.csv``: one crop per photo around all final boxes;
- ``box/roi_manifest.csv``: one crop per final box.

The crops use T03's 8% padding and rounding, ``crop_index`` and file names. Rows copy the photo's
manifest columns (case, split and label are inherited), plus ``fallback`` (none / top_box /
full_image), ``auto_score`` and the unpadded ``box_*`` coordinates. A ``full_image`` crop is the
whole photo. With ``--image-root`` the crops are cut from the EXIF-oriented photos; without it only
the manifests are written (dry run).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.create_roi_dataset import safe_folder  # noqa: E402
from src.detection.boxes import COORDINATES, PADDING, candidate_boxes, final_boxes, padded_crop_box  # noqa: E402
from src.detection.common import PHOTO, load_photo, photo_table  # noqa: E402
from src.detection.evaluate_detection import load_detections  # noqa: E402
from src.training.data import GROUP_COLUMN, LABEL_COLUMN  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402


DESIGNS = ("union", "box")


def final_boxes_by_photo(photos: pd.DataFrame, detections: pd.DataFrame, threshold: float,
                         limits: Dict) -> Dict[str, tuple]:
    # For each photo: its raw boxes (none if it had no detection), the step-1 rules, then threshold, cap and fallback
    by_photo = {p: g for p, g in detections.groupby(PHOTO, sort=False)}
    result = {}
    for row in photos.to_dict("records"):
        rows = by_photo.get(row[PHOTO])
        boxes = rows[COORDINATES].to_numpy(float) if rows is not None else np.zeros((0, 4))
        scores = rows["score"].to_numpy(float) if rows is not None else np.zeros(0)
        w, h = row["source_width"], row["source_height"]
        cand = candidate_boxes(boxes, scores, w, h, limits["max_box_share"])
        result[row[PHOTO]] = final_boxes(*cand, w, h, threshold, int(limits["max_boxes_per_photo"]))
    return result


def build_rows(photos: pd.DataFrame, finals: Dict[str, tuple], arm: str) -> Dict[str, List[Dict]]:
    rows: Dict[str, List[Dict]] = {d: [] for d in DESIGNS}
    for photo in photos.to_dict("records"):
        # This photo's final boxes and the folder and file names its crops get (same pattern as T03)
        boxes, scores, fallback = finals[photo[PHOTO]]
        w, h = photo["source_width"], photo["source_height"]
        stem = Path(str(photo["filename"]).strip()).stem
        folder = f"{photo['split']}/{safe_folder(photo[LABEL_COLUMN])}"
        # Columns shared by every crop of the photo: case, split, label, arm, fallback
        base = {**photo, "detector_arm": arm, "fallback": fallback, "padding_ratio": PADDING,
                "source_width": w, "source_height": h}
        # Union design: one crop around all boxes. Per-box design: one crop per box
        groups = {"union": [(1, boxes, f"{stem}_roi")],
                  "box": [(k, boxes[[k - 1]], f"{stem}_bbox{k:02d}") for k in range(1, len(boxes) + 1)]}
        for design, crops in groups.items():
            for crop_index, crop_boxes, name in crops:
                # No box at all: the crop is the whole photo; otherwise pad the box(es) by 8%
                if fallback == "full_image":
                    x1, y1, x2, y2 = 0, 0, w, h
                else:
                    x1, y1, x2, y2 = padded_crop_box(crop_boxes, w, h)
                # The crop's detector score (union: the best of its boxes; full image: none)
                score = scores[crop_index - 1] if design == "box" else (float(np.max(scores)) if np.isfinite(scores).all() else np.nan)
                rows[design].append({
                    **base, "roi_relative_path": f"{folder}/{name}.jpg",
                    "crop_mode": "union" if design == "union" else "individual",
                    "crop_index": crop_index, "bbox_count": len(crop_boxes), "auto_score": score,
                    **({"box_x1": crop_boxes[0, 0], "box_y1": crop_boxes[0, 1], "box_x2": crop_boxes[0, 2],
                        "box_y2": crop_boxes[0, 3]} if design == "box" else {}),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-box union and per-box crops of one detector arm (T19).")
    parser.add_argument("--selection", type=Path, required=True, help="The arm's frozen selection.json.")
    parser.add_argument("--detections", type=Path, nargs="+", required=True,
                        help="Detections files of the chosen configuration, together covering every photo once.")
    parser.add_argument("--output-dir", type=Path, required=True, help="data/roi_auto/<arm>")
    parser.add_argument("--image-root", type=Path, default=None, help="Raw photo tree; omit for a dry run.")
    parser.add_argument("--crop-root", type=Path, default=None,
                        help="Where crops go: <crop-root>_union/ and <crop-root>_box/ (live run only).")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--limits", type=Path, default=DATA_DIR / "roi_auto/reference_limits.json")
    args = parser.parse_args(argv)
    if (args.image_root is None) != (args.crop_root is None):
        parser.error("--image-root and --crop-root go together")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Only an arm that passed its gate makes crops, with the same limits it was selected under
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if not selection.get("gate_passed", False):
        raise SystemExit(f"{args.selection}: the arm failed its recall gate; it makes no crops (T19 §2.4)")
    limits = json.loads(args.limits.read_text(encoding="utf-8"))
    if {k: limits[k] for k in selection["limits"]} != selection["limits"]:
        raise SystemExit(f"{args.limits} differs from the limits frozen in {args.selection}")

    # Read every detections file once; each photo must come from exactly one file
    resolved = [p.resolve() for p in args.detections]
    if len(set(resolved)) != len(resolved):
        raise SystemExit("A detections file is listed more than once")
    parts = []
    for path in resolved:
        frame = load_detections(path)
        frame["detections_file"] = str(path)
        parts.append(frame)
    detections = pd.concat(parts, ignore_index=True)
    files_per_photo = detections.groupby(PHOTO)["detections_file"].nunique()
    if (files_per_photo > 1).any():
        raise SystemExit(f"Photos in more than one detections file, e.g. {list(files_per_photo[files_per_photo > 1].index[:3])}")

    # All 1,100 photos; every detected photo must be one of them
    photos = photo_table(args.box_manifest)
    unknown = set(detections[PHOTO]) - set(photos[PHOTO])
    if unknown:
        raise SystemExit(f"{len(unknown)} detected photos are not in {args.box_manifest}, e.g. {sorted(unknown)[:3]}")
    # Final boxes per photo, then the manifest rows of both designs
    finals = final_boxes_by_photo(photos, detections, float(selection["threshold"]), limits)
    rows = build_rows(photos, finals, selection["arm"])

    # Summary: how many photos needed a fallback, overall and per split
    summary = {"arm": selection["arm"], "chosen_config": selection["chosen_config"], "threshold": selection["threshold"],
               "detections": [str(p) for p in args.detections], "dry_run": args.image_root is None,
               "photos": int(len(photos)), "fallback": dict(Counter(f for _, _, f in finals.values())),
               "fallback_by_split": {s: dict(Counter(finals[p][2] for p in photos.loc[photos["split"] == s, PHOTO]))
                                     for s in ("train", "validation", "test")}}
    # Write one manifest per design (training and prediction read these)
    for design in DESIGNS:
        frame = pd.DataFrame(rows[design])
        out = args.output_dir / design
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "roi_manifest.csv", index=False, float_format="%.10g")
        summary[f"{design}_crops"] = int(len(frame))
        summary[f"{design}_crops_by_split"] = frame["split"].value_counts().sort_index().to_dict()

    # Live run: open each photo once and save all its crops as JPEG (as T03 did)
    if args.image_root is not None:
        for row in photos.to_dict("records"):
            image = load_photo(args.image_root, row)
            for design in DESIGNS:
                for crop in (r for r in rows[design] if r[PHOTO] == row[PHOTO]):
                    path = Path(f"{args.crop_root}_{design}") / crop["roi_relative_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop((crop["x1"], crop["y1"], crop["x2"], crop["y2"])).save(path, format="JPEG", quality=95)
    # Write the summary and one log line
    (args.output_dir / "crop_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{selection['arm']}: {summary['union_crops']} union and {summary['box_crops']} per-box crops; "
          f"fallbacks {summary['fallback']}. Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
