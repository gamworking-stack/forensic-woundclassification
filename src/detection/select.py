#!/usr/bin/env python3
"""Choose one detector arm's configuration and threshold, and judge its recall gate (T19 §2.2–§2.4).

Input: the arm's detections files, one per configuration (``<config>.csv``, see
``evaluate_detection``), and the splits the arm selects on: ``validation`` for 4F, ``train`` and
``validation`` for 4Z and 4P. **Test photos are never read**: detections of test photos are dropped
before anything is computed, and their count is recorded.

For every configuration it computes class-agnostic AP@0.5 on the selection photos. The winner is
the highest AP@0.5; ties go to the configuration listed first in ``--configs`` (callers list them
from simplest to most complex, as T19 fixes for each arm). For the winner it chooses the F1
threshold and judges the gate. Everything goes to ``selection.json``, which every later step reads
and which ``evaluate_detection`` needs before it touches test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.evaluate_detection import (  # noqa: E402
    GATE_RECALL,
    PHOTO,
    candidates,
    choose_threshold,
    coco_ap,
    coco_images,
    gate_passed,
    load_detections,
    reference_photos,
)
from src.training.common import git_state  # noqa: E402
from src.training.data import GROUP_COLUMN  # noqa: E402


ARMS = ("4Z", "4P", "4F")
SELECTION_SPLITS = {"4Z": ("train", "validation"), "4P": ("train", "validation"), "4F": ("validation",)}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(arm: str, configs: List[str], detections_dir: Path, reference: pd.DataFrame,
           limits: Dict[str, Any]) -> Dict[str, Any]:
    """Score every configuration on the arm's selection photos; choose, threshold and gate."""
    splits = SELECTION_SPLITS[arm]
    photos = reference_photos(reference, splits)
    test_photos = set(reference.loc[reference["split"] == "test", PHOTO])
    scores, dropped_test, hashes = [], {}, {}
    cands_of = {}
    for config in configs:
        path = detections_dir / f"{config}.csv"
        detections = load_detections(path)
        hashes[config] = file_sha256(path)
        is_test = detections[PHOTO].isin(test_photos)
        dropped_test[config] = int(is_test.sum())
        detections = detections[~is_test]
        cands_of[config] = candidates(detections, photos, limits["max_box_share"])
        ap = coco_ap(coco_images(photos, cands_of[config]))
        scores.append({"config": config, **ap})
    best = max(range(len(configs)), key=lambda i: (scores[i]["ap50"], -i))  # ties: earlier in --configs
    chosen = configs[best]
    threshold = choose_threshold(photos, cands_of[chosen], int(limits["max_boxes_per_photo"]))
    return {
        "arm": arm,
        "selection_splits": list(splits),
        "selection_photos": int(len(photos)),
        "selection_cases": int(photos["case"].nunique()),
        "configs": scores,
        "chosen_config": chosen,
        "threshold": threshold["threshold"],
        "threshold_detail": threshold,
        "gate_recall": GATE_RECALL,
        "gate_passed": gate_passed(threshold["recall"]),
        "test_detections_dropped": dropped_test,
        "detections_sha256": hashes,
        "limits": {k: limits[k] for k in ("max_box_share", "max_boxes_per_photo", "size_bins")},
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one detector arm's configuration and threshold (T19).")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--detections-dir", type=Path, required=True,
                        help="Folder with one <config>.csv detections file per configuration.")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Configuration names, simplest first (ties go to the earlier one).")
    parser.add_argument("--output", type=Path, required=True, help="selection.json to write (never overwritten).")
    parser.add_argument("--reference", type=Path, default=Path("data/roi_auto/reference_boxes.csv"))
    parser.add_argument("--limits", type=Path, default=Path("data/roi_auto/reference_limits.json"))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} exists: a selection is frozen once written (T19 §2.6)")
    reference = pd.read_csv(args.reference, dtype={PHOTO: str, GROUP_COLUMN: str})
    limits = json.loads(args.limits.read_text(encoding="utf-8"))
    result = select(args.arm, args.configs, args.detections_dir, reference, limits)
    result.update({"detections_dir": str(args.detections_dir), **git_state(),
                   "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    d = result["threshold_detail"]
    verdict = "PASSED" if result["gate_passed"] else "FAILED"
    print(f"{args.arm}: chose {result['chosen_config']} (AP50 "
          f"{next(s['ap50'] for s in result['configs'] if s['config'] == result['chosen_config']):.3f}); "
          f"threshold {result['threshold']:.4f}, F1 {d['f1']:.3f}, recall {d['recall']:.3f}; gate {verdict}. "
          f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
