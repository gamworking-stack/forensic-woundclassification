#!/usr/bin/env python3
"""Detection metrics for the auto-box ROI experiment (T19 §2.3, §2.6): auto boxes against manual boxes.

Every function works on one detector configuration's **detections file**: one row per raw box with
``relative_path`` (the photo), ``score`` and ``x1, y1, x2, y2`` in source pixels. A photo with no row
had no detection. The reference is ``data/roi_auto/reference_boxes.csv`` and the rules come from
``data/roi_auto/reference_limits.json`` (oversize share, box cap, size bins).

- ``candidates``: T19 §2.3 step 1 per photo (oversize rule, NMS, top 100). **mAP is computed here.**
- ``coco_ap``: COCO evaluation with ``pycocotools``, one category; AP@0.5 is the primary metric.
  Size bins are shares of the image area (training-split tertiles), not COCO's pixel areas, which
  would put every box of a 4000 px photo in "large".
- ``choose_threshold``: the score that maximises pooled box F1 at IoU >= 0.5 (step 2), with the
  box cap applied and **no fallback**; ties go to the higher threshold. Its recall is the one the
  recall gate (``GATE_RECALL``) is judged on.
- ``finalize`` and ``photo_results``: step 3 (threshold, cap, fallback) and per-photo counts, from
  which the operating-point and photo-level metrics are computed.
- ``bootstrap``: case-level percentile intervals, drawn exactly as ``src/evaluation/metrics.py`` does.

As a script it evaluates one split from a frozen ``selection.json`` (``threshold``, ``gate_passed``) and
refuses the test split for an arm that failed its gate.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.boxes import (  # noqa: E402
    COORDINATES,
    MATCH_IOU,
    area_share,
    as_boxes,
    candidate_boxes,
    final_boxes,
    greedy_match,
)
from src.training.data import GROUP_COLUMN, LABEL_COLUMN  # noqa: E402
from src.training.photo import MANIFEST_PHOTO_COLUMN  # noqa: E402


PHOTO = MANIFEST_PHOTO_COLUMN
GATE_RECALL = 0.50
DETECTION_COLUMNS = [PHOTO, "score", *COORDINATES]
BOOTSTRAP_METRICS = ("ap50", "precision", "recall", "photo_any_found", "photo_all_found")

Candidates = Dict[str, Tuple[np.ndarray, np.ndarray]]


# ---------- inputs ----------

def reference_photos(reference: pd.DataFrame, splits: Sequence[str]) -> pd.DataFrame:
    """One row per photo of ``splits``: case, label, size and its manual boxes as an ``(N, 4)`` array."""
    # Keep the reference boxes of the requested splits
    rows = reference[reference["split"].isin(splits)]
    if rows.empty:
        raise ValueError(f"No reference boxes in splits {list(splits)}")
    # One row per photo with its case, wound type, split and size
    photos = rows.groupby(PHOTO, sort=True).agg(
        case=(GROUP_COLUMN, "first"), label=(LABEL_COLUMN, "first"), split=("split", "first"),
        width=("source_width", "first"), height=("source_height", "first"))
    # Attach each photo's manual boxes as an (N, 4) array
    photos["truth"] = [as_boxes(g[COORDINATES].to_numpy(float)) for _, g in rows.groupby(PHOTO, sort=True)]
    photos["case"] = photos["case"].astype(str)
    return photos


def load_detections(path: Path) -> pd.DataFrame:
    # Read the file and check it has the columns, finite numbers and valid boxes
    detections = pd.read_csv(path, dtype={PHOTO: str})
    missing = set(DETECTION_COLUMNS) - set(detections.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    values = detections[["score", *COORDINATES]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite scores or coordinates")
    as_boxes(detections[COORDINATES].to_numpy(float))
    return detections


def candidates(detections: pd.DataFrame, photos: pd.DataFrame, max_share: float) -> Candidates:
    """T19 §2.3 step 1 for every photo of ``photos``; detections of other photos are ignored."""
    # Detections grouped by photo; a photo without rows had no detection
    by_photo = {p: g for p, g in detections.groupby(PHOTO, sort=False)}
    result: Candidates = {}
    for photo, info in photos.iterrows():
        # This photo's raw boxes and scores (empty if none), then the step-1 rules
        rows = by_photo.get(photo)
        boxes = rows[COORDINATES].to_numpy(float) if rows is not None else np.zeros((0, 4))
        scores = rows["score"].to_numpy(float) if rows is not None else np.zeros(0)
        result[photo] = candidate_boxes(boxes, scores, info["width"], info["height"], max_share)
    return result


# ---------- COCO average precision ----------

def coco_ap(images: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]],
            size_bins: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """COCO AP for one category over ``images`` = ``(truth, boxes, scores, width, height)``.

    The same photo may appear more than once (bootstrap); every occurrence is its own image. With
    ``size_bins`` = ``[a, b]``, small / medium / large are boxes with image-area share < a, a..b, >= b.
    Returns ``ap`` (0.5:0.95), ``ap50``, ``ap75`` and ``ap_small/medium/large`` (NaN when undefined).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Build COCO's input: every image, its manual boxes (ground truth) and the auto boxes with scores
    gt_images, gt_annotations, detections = [], [], []
    for image_id, (truth, boxes, scores, width, height) in enumerate(images, start=1):
        gt_images.append({"id": image_id, "width": float(width), "height": float(height)})
        for box in as_boxes(truth):
            gt_annotations.append({
                "id": len(gt_annotations) + 1, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "area": _area(box, width, height, size_bins),
            })
        for box, score in zip(as_boxes(boxes), scores):
            detections.append({"image_id": image_id, "category_id": 1, "score": float(score),
                               "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                               "_area": _area(box, width, height, size_bins)})
    # No auto box at all: AP is 0 and the size APs are undefined
    names = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")
    if not gt_annotations:
        raise ValueError("COCO evaluation needs at least one reference box")
    if not detections:
        return {n: (0.0 if n in ("ap", "ap50", "ap75") else float("nan")) for n in names}

    with contextlib.redirect_stdout(io.StringIO()):
        # Load the manual boxes into a COCO object and the auto boxes as its results
        truth_coco = COCO()
        truth_coco.dataset = {"images": gt_images, "annotations": gt_annotations,
                              "categories": [{"id": 1, "name": "wound"}]}
        truth_coco.createIndex()
        found = truth_coco.loadRes([{k: v for k, v in d.items() if k != "_area"} for d in detections])
        # Size bins by image share: overwrite the auto boxes' area with their share
        if size_bins is not None:
            for annotation_id, detection in zip(sorted(found.anns), detections):
                found.anns[annotation_id]["area"] = detection["_area"]
        # Run COCO's evaluation (with our size bins if given); output printing is silenced
        evaluator = COCOeval(truth_coco, found, iouType="bbox")
        if size_bins is not None:
            low, high = size_bins
            evaluator.params.areaRng = [[0.0, 1e9], [0.0, low], [low, high], [high, 1e9]]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    # COCO reports -1 when a value is undefined; turn that into NaN
    stats = [float(v) if v >= 0 else float("nan") for v in evaluator.stats[:6]]
    return dict(zip(names, stats))


def _area(box: np.ndarray, width: float, height: float, size_bins: Optional[Sequence[float]]) -> float:
    """COCO's area field: the image-area share with size bins, else pixels as COCO expects."""
    if size_bins is None:
        return float((box[2] - box[0]) * (box[3] - box[1]))
    return float(area_share(box, width, height)[0])


def coco_images(photos: pd.DataFrame, cands: Candidates) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    return [(info["truth"], *cands[photo], info["width"], info["height"]) for photo, info in photos.iterrows()]


# ---------- operating threshold (step 2) ----------

def ranked_hits(photos: pd.DataFrame, cands: Candidates, max_boxes: int) -> pd.DataFrame:
    """Every capped candidate with its score and whether it matches a manual box.

    Greedy matching handles predictions by descending score, so a box's match does not depend on
    any lower-scoring box: the matches at any threshold are a prefix of this one run.
    """
    rows = []
    # For each photo: its capped candidates, and which of them match a manual box
    for photo, info in photos.iterrows():
        boxes, scores = cands[photo]
        boxes, scores = boxes[:max_boxes], scores[:max_boxes]  # candidates come in descending score
        hits = np.zeros(len(boxes), dtype=bool)
        for p, _, _ in greedy_match(boxes, scores, info["truth"], MATCH_IOU):
            hits[p] = True
        rows += [{PHOTO: photo, "score": s, "hit": h} for s, h in zip(scores, hits)]
    return pd.DataFrame(rows, columns=[PHOTO, "score", "hit"])


def choose_threshold(photos: pd.DataFrame, cands: Candidates, max_boxes: int) -> Dict[str, float]:
    """The score maximising pooled F1 of boxes with ``score >= threshold`` (cap applied, no fallback).

    Every distinct candidate score is tried; ties in F1 go to the higher threshold. The returned
    ``recall`` is the one the recall gate uses.
    """
    # Total manual boxes, and every candidate with its hit/miss
    n_true = int(sum(len(t) for t in photos["truth"]))
    hits = ranked_hits(photos, cands, max_boxes)
    if hits.empty:
        return {"threshold": float("nan"), "f1": 0.0, "precision": float("nan"), "recall": 0.0,
                "n_predicted": 0, "n_matched": 0, "n_true": n_true}
    # Sort all candidates by score; running sums give matches and predictions at every threshold
    order = np.argsort(-hits["score"].to_numpy(), kind="stable")
    scores = hits["score"].to_numpy()[order]
    matched = np.cumsum(hits["hit"].to_numpy()[order])
    predicted = np.arange(1, len(order) + 1)
    last_of_score = np.r_[scores[1:] != scores[:-1], True]  # each distinct score's last position
    # F1 at each distinct score; pick the best (the first, highest threshold, on ties)
    f1 = 2 * matched / (predicted + n_true)
    candidates_at = np.flatnonzero(last_of_score)
    best = candidates_at[np.argmax(f1[candidates_at])]  # argmax takes the first: the highest threshold
    return {"threshold": float(scores[best]), "f1": float(f1[best]),
            "precision": float(matched[best] / predicted[best]), "recall": float(matched[best] / n_true),
            "n_predicted": int(predicted[best]), "n_matched": int(matched[best]), "n_true": n_true}


def gate_passed(recall: float) -> bool:
    return bool(recall >= GATE_RECALL)


# ---------- operating point (step 3) ----------

def finalize(photos: pd.DataFrame, cands: Candidates, threshold: float,
             max_boxes: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    return {photo: final_boxes(*cands[photo], info["width"], info["height"], threshold, max_boxes)
            for photo, info in photos.iterrows()}


def photo_results(photos: pd.DataFrame, finals: Dict[str, Tuple[np.ndarray, np.ndarray, str]]) -> pd.DataFrame:
    """Per photo: manual boxes, final auto boxes, matches, summed IoU of matches and the fallback used."""
    rows = []
    # Match each photo's final boxes to its manual boxes and count them
    for photo, info in photos.iterrows():
        boxes, scores, fallback = finals[photo]
        matches = greedy_match(boxes, scores, info["truth"], MATCH_IOU)
        rows.append({PHOTO: photo, GROUP_COLUMN: info["case"], LABEL_COLUMN: info["label"],
                     "split": info["split"], "n_true": len(info["truth"]), "n_predicted": len(boxes),
                     "n_matched": len(matches), "iou_sum": float(sum(i for _, _, i in matches)),
                     "fallback": fallback})
    return pd.DataFrame(rows)


def operating_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """Pooled box metrics and photo-level rates from ``photo_results`` rows."""
    # Pooled counts over all photos give precision, recall and F1
    matched, predicted, true = (int(results[c].sum()) for c in ("n_matched", "n_predicted", "n_true"))
    precision = matched / predicted if predicted else float("nan")
    recall = matched / true if true else float("nan")
    f1 = 2 * matched / (predicted + true) if predicted + true else float("nan")
    # Photo-level rates: fallbacks, at least one wound found, every wound found
    photo_recall = results["n_matched"] / results["n_true"]
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": float(results["iou_sum"].sum() / matched) if matched else float("nan"),
        "share_no_box": float((results["fallback"] != "none").mean()),
        "share_full_image": float((results["fallback"] == "full_image").mean()),
        "photo_any_found": float((results["n_matched"] > 0).mean()),
        "photo_all_found": float((results["n_matched"] == results["n_true"]).mean()),
        "photo_mean_recall": float(photo_recall.mean()),
        "n_photos": int(len(results)), "n_true": true, "n_predicted": predicted, "n_matched": matched,
    }


# ---------- case bootstrap ----------

def bootstrap(photos: pd.DataFrame, cands: Candidates, results: pd.DataFrame, n_resamples: int = 2000,
              seed: int = 42, level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """Percentile intervals of ``BOOTSTRAP_METRICS``; cases drawn with replacement, a case's photos together.

    Drawn as in ``src/evaluation/metrics.py``: sorted case ids, ``default_rng(seed).integers``.
    """
    # Same draw as the classification bootstrap: cases sorted, resampled with replacement
    rng = np.random.default_rng(seed)
    order = list(photos.index)
    results = results.set_index(PHOTO).loc[order]
    case_ids = np.array(sorted(set(photos["case"])))
    rows_of = [np.flatnonzero(photos["case"].to_numpy() == c) for c in case_ids]
    # Precompute COCO inputs and per-photo counts once, so each resample only re-sums them
    images = coco_images(photos, cands)
    counts = results[["n_matched", "n_predicted", "n_true"]].to_numpy(float)
    draws: Dict[str, List[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(n_resamples):
        # One resample: draw cases, take all their photos, recompute every metric
        picked = rng.integers(0, len(case_ids), size=len(case_ids))
        idx = np.concatenate([rows_of[i] for i in picked])
        matched, predicted, true = counts[idx].sum(axis=0)
        draws["ap50"].append(coco_ap([images[i] for i in idx])["ap50"])
        draws["precision"].append(matched / predicted if predicted else np.nan)
        draws["recall"].append(matched / true)
        draws["photo_any_found"].append(float((counts[idx, 0] > 0).mean()))
        draws["photo_all_found"].append(float((counts[idx, 0] == counts[idx, 2]).mean()))
    # 95% percentile interval of each metric across resamples
    tail = (1.0 - level) / 2 * 100
    out = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        out[name] = {"low": float(np.percentile(values, tail)), "high": float(np.percentile(values, 100 - tail)),
                     "n_resamples": int(len(values))}
    return out


def evaluate(photos: pd.DataFrame, cands: Candidates, threshold: float, max_boxes: int,
             size_bins: Optional[Sequence[float]], n_resamples: int, seed: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """All T19 §2.6 metrics on ``photos``: COCO AP, operating point, photo level, and their intervals."""
    # Final boxes at the operating threshold, per-photo counts, COCO AP, then the intervals
    finals = finalize(photos, cands, threshold, max_boxes)
    results = photo_results(photos, finals)
    metrics = {**coco_ap(coco_images(photos, cands), size_bins), **operating_metrics(results)}
    intervals = bootstrap(photos, cands, results, n_resamples, seed) if n_resamples else {}
    return {"metrics": metrics, "intervals": intervals}, results


# ---------- script ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection metrics of one detector configuration (T19 §2.6).")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True,
                        help="Frozen selection.json of the arm: needs 'threshold' and 'gate_passed'.")
    parser.add_argument("--split", choices=("train", "validation", "test"), nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("data/roi_auto/reference_boxes.csv"))
    parser.add_argument("--limits", type=Path, default=Path("data/roi_auto/reference_limits.json"))
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # An arm that failed its gate must never be scored on test
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if "test" in args.split and not selection.get("gate_passed", False):
        raise SystemExit(f"{args.selection}: the arm did not pass its recall gate, so it never touches test (T19 §2.4)")
    # Load the limits, the photos of the split and the step-1 candidates
    limits = json.loads(args.limits.read_text(encoding="utf-8"))
    photos = reference_photos(pd.read_csv(args.reference, dtype={PHOTO: str, GROUP_COLUMN: str}), args.split)
    cands = candidates(load_detections(args.detections), photos, limits["max_box_share"])
    threshold, cap, bins = float(selection["threshold"]), int(limits["max_boxes_per_photo"]), limits["size_bins"]

    report: Dict[str, Any] = {"detections": str(args.detections), "selection": str(args.selection),
                              "split": args.split, "threshold": threshold, "max_boxes_per_photo": cap,
                              "size_bins": bins, "n_resamples": args.n_resamples, "seed": args.seed}
    # Metrics for all photos, then for each wound type on its own
    report["overall"], results = evaluate(photos, cands, threshold, cap, bins, args.n_resamples, args.seed)
    by_type = {}
    for label in sorted(photos["label"].unique()):
        subset = photos[photos["label"] == label]
        by_type[label], _ = evaluate(subset, cands, threshold, cap, bins, args.n_resamples, args.seed)
    report["by_type"] = by_type

    # Write the metrics, the per-photo results and every final box
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "detection_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    results.to_csv(args.output_dir / "photo_results.csv", index=False)
    finals = finalize(photos, cands, threshold, cap)
    rows = [{PHOTO: photo, "box_index": k, **dict(zip(COORDINATES, box)), "score": score, "fallback": fallback}
            for photo, (boxes, scores, fallback) in finals.items()
            for k, (box, score) in enumerate(zip(boxes, scores), start=1)]
    pd.DataFrame(rows).to_csv(args.output_dir / "final_boxes.csv", index=False, float_format="%.10g")
    m = report["overall"]["metrics"]
    print(f"{'+'.join(args.split)}: AP50 {m['ap50']:.3f}, precision {m['precision']:.3f}, recall {m['recall']:.3f}, "
          f"photos with a wound found {m['photo_any_found']:.3f}, no box {m['share_no_box']:.3f}. "
          f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
