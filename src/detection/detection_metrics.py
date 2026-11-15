#!/usr/bin/env python3
"""T37: the full detection evaluation of one detector arm on test, and the comparison of arms.

Built on T19's rules (``evaluate_detection.py``): the same step-1 candidates, frozen threshold, box cap,
fallback and greedy one-to-one matching at IoU >= 0.5. One class (wound). It adds what a medical
detection report usually shows:

- **IoU**: mean IoU of matched pairs (T19), and the mean over manual boxes of the best IoU any final
  box reaches (0 when none overlaps), which also counts the missed wounds;
- **operating point** (final boxes at the frozen threshold): precision, recall (sensitivity), F1,
  false positives per image;
- **AP50, AP75, mAP50-95** (COCO, 101 points, maxDets 100; with one class mAP is the AP) and AP by
  box size (image-share tertiles, T19);
- **FROC**: sensitivity against false positives per image as the threshold sweeps the capped
  candidates (no fallback); sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 FP per image (the highest
  sensitivity reached without exceeding that FP rate) and their mean (the FROC score);
- **precision-recall curve** from the same sweep.

Intervals: 2,000 bootstrap resamples, seed 42, drawn as T19 draws them; with ``--patient-cases`` the
resampled unit is the patient (T25), otherwise the T23 group. Overall and per wound type.

``evaluate`` writes ``metrics.json``, ``froc.csv`` and ``photo_results.csv`` (per photo, so the output
folder stays outside the repository). ``report`` combines several arms into
``detection_metrics_table.csv``, ``detection_by_type.csv`` and ``froc_pr.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.boxes import MATCH_IOU, as_boxes, greedy_match, iou_matrix  # noqa: E402
from src.detection.evaluate_detection import (  # noqa: E402
    PHOTO,
    Candidates,
    candidates,
    coco_ap,
    coco_images,
    finalize,
    load_detections,
    operating_metrics,
    ranked_hits,
    reference_photos,
)
from src.training.data import GROUP_COLUMN, LABEL_COLUMN  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
FP_LEVELS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BOOTSTRAP_METRICS = ("ap50", "ap75", "ap", "precision", "recall", "f1", "fp_per_image", "mean_iou",
                     "mean_best_iou", "sensitivity_at_1fp", "froc_score", "photo_any_found")
ARM_LABELS = {"4Y": "YOLO11 (fine-tuned)", "4F": "Faster R-CNN (fine-tuned)", "4Z": "Grounding DINO (zero-shot)",
              "4P": "Public YOLOv8"}

Hits = List[Tuple[np.ndarray, np.ndarray]]


def fp_key(level: float) -> str:
    return f"sensitivity_at_{level:g}fp"


# ---------- per photo ----------

def with_patients(photos: pd.DataFrame, patient_cases: Path) -> pd.DataFrame:
    """``photos`` with ``case`` replaced by the photo's patient (T25)."""
    patients = pd.read_csv(patient_cases, dtype=str).set_index(PHOTO)["patient_case"]
    missing = sorted(set(photos.index) - set(patients.index))
    if missing:
        raise ValueError(f"{len(missing)} photos have no patient case, e.g. {missing[:3]}")
    return photos.assign(case=patients.loc[photos.index].to_numpy())


def photo_hits(photos: pd.DataFrame, cands: Candidates, max_boxes: int) -> Hits:
    """Per photo (in ``photos`` order): the capped candidates' scores and whether each matches a manual box."""
    hits = ranked_hits(photos, cands, max_boxes)
    by_photo = {p: g for p, g in hits.groupby(PHOTO, sort=False)}
    out = []
    for photo in photos.index:
        g = by_photo.get(photo)
        out.append((np.zeros(0), np.zeros(0, dtype=bool)) if g is None
                   else (g["score"].to_numpy(float), g["hit"].to_numpy(bool)))
    return out


def photo_table_of(photos: pd.DataFrame, finals: Dict[str, Tuple[np.ndarray, np.ndarray, str]]) -> pd.DataFrame:
    """T19's per-photo counts plus the summed best IoU of every manual box (0 where nothing overlaps)."""
    rows = []
    for photo, info in photos.iterrows():
        boxes, scores, fallback = finals[photo]
        truth = as_boxes(info["truth"])
        matches = greedy_match(boxes, scores, truth, MATCH_IOU)
        best = iou_matrix(truth, boxes).max(axis=1) if len(boxes) and len(truth) else np.zeros(len(truth))
        rows.append({PHOTO: photo, "case": info["case"], LABEL_COLUMN: info["label"], "split": info["split"],
                     "n_true": len(truth), "n_predicted": len(boxes), "n_matched": len(matches),
                     "iou_sum": float(sum(i for _, _, i in matches)), "best_iou_sum": float(best.sum()),
                     "fallback": fallback})
    return pd.DataFrame(rows)


# ---------- curves ----------

def sweep(hits: Hits, n_true: int, n_photos: int) -> pd.DataFrame:
    """FROC and precision-recall points at every distinct score, highest first, from threshold +inf."""
    scores = np.concatenate([s for s, _ in hits]) if hits else np.zeros(0)
    matched_flags = np.concatenate([h for _, h in hits]) if hits else np.zeros(0, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    scores, flags = scores[order], matched_flags[order]
    matched = np.cumsum(flags)
    predicted = np.arange(1, len(scores) + 1)
    last = np.r_[scores[1:] != scores[:-1], True] if len(scores) else np.zeros(0, dtype=bool)
    idx = np.flatnonzero(last)
    curve = pd.DataFrame({"threshold": scores[idx], "n_predicted": predicted[idx], "n_matched": matched[idx]})
    curve = pd.concat([pd.DataFrame({"threshold": [np.inf], "n_predicted": [0], "n_matched": [0]}), curve],
                      ignore_index=True)
    curve["sensitivity"] = curve["n_matched"] / n_true if n_true else np.nan
    curve["fp_per_image"] = (curve["n_predicted"] - curve["n_matched"]) / n_photos
    curve["precision"] = np.where(curve["n_predicted"] > 0, curve["n_matched"] / curve["n_predicted"].clip(lower=1), np.nan)
    return curve


def froc_points(curve: pd.DataFrame, levels: Sequence[float] = FP_LEVELS) -> Dict[str, float]:
    """Highest sensitivity with at most ``level`` FP per image, per level, and their mean (FROC score)."""
    out = {}
    for level in levels:
        ok = curve.loc[curve["fp_per_image"] <= level + 1e-12, "sensitivity"]
        out[fp_key(level)] = float(ok.max()) if len(ok) else 0.0
    out["froc_score"] = float(np.mean([out[fp_key(level)] for level in levels]))
    return out


# ---------- metrics ----------

def point_metrics(results: pd.DataFrame, coco: Dict[str, float], curve: pd.DataFrame) -> Dict[str, float]:
    base = operating_metrics(results)
    n_true = int(results["n_true"].sum())
    return {**coco, **base,
            "fp_per_image": float((results["n_predicted"] - results["n_matched"]).sum() / len(results)),
            "mean_best_iou": float(results["best_iou_sum"].sum() / n_true) if n_true else float("nan"),
            **froc_points(curve)}


def bootstrap(photos: pd.DataFrame, cands: Candidates, results: pd.DataFrame, hits: Hits,
              n_resamples: int, seed: int, level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """Percentile intervals of ``BOOTSTRAP_METRICS``; cases (``photos['case']``) drawn as T19 draws them."""
    rng = np.random.default_rng(seed)
    case_ids = np.array(sorted(set(photos["case"])))
    rows_of = [np.flatnonzero(photos["case"].to_numpy() == c) for c in case_ids]
    images = coco_images(photos, cands)
    results = results.set_index(PHOTO).loc[list(photos.index)]
    counts = results[["n_matched", "n_predicted", "n_true", "iou_sum", "best_iou_sum"]].to_numpy(float)
    draws: Dict[str, List[float]] = {m: [] for m in BOOTSTRAP_METRICS}
    for _ in range(n_resamples):
        picked = rng.integers(0, len(case_ids), size=len(case_ids))
        idx = np.concatenate([rows_of[i] for i in picked])
        matched, predicted, true, iou_sum, best_sum = counts[idx].sum(axis=0)
        coco = coco_ap([images[i] for i in idx])
        curve = sweep([hits[i] for i in idx], int(true), len(idx))
        froc = froc_points(curve)
        draws["ap50"].append(coco["ap50"])
        draws["ap75"].append(coco["ap75"])
        draws["ap"].append(coco["ap"])
        draws["precision"].append(matched / predicted if predicted else np.nan)
        draws["recall"].append(matched / true)
        draws["f1"].append(2 * matched / (predicted + true))
        draws["fp_per_image"].append((predicted - matched) / len(idx))
        draws["mean_iou"].append(iou_sum / matched if matched else np.nan)
        draws["mean_best_iou"].append(best_sum / true)
        draws["sensitivity_at_1fp"].append(froc[fp_key(1.0)])
        draws["froc_score"].append(froc["froc_score"])
        draws["photo_any_found"].append(float((counts[idx, 0] > 0).mean()))
    tail = (1.0 - level) / 2 * 100
    out = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        out[name] = {"low": float(np.percentile(values, tail)), "high": float(np.percentile(values, 100 - tail)),
                     "n_resamples": int(len(values))}
    return out


def evaluate_photos(photos: pd.DataFrame, cands: Candidates, threshold: float, max_boxes: int,
                    size_bins: Optional[Sequence[float]], n_resamples: int, seed: int
                    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Every metric on ``photos`` with intervals; also the per-photo table and the FROC/PR curve."""
    results = photo_table_of(photos, finalize(photos, cands, threshold, max_boxes))
    hits = photo_hits(photos, cands, max_boxes)
    curve = sweep(hits, int(results["n_true"].sum()), len(results))
    metrics = point_metrics(results, coco_ap(coco_images(photos, cands), size_bins), curve)
    metrics.update({"n_cases": int(photos["case"].nunique()), "threshold": float(threshold)})
    intervals = bootstrap(photos, cands, results, hits, n_resamples, seed) if n_resamples else {}
    return {"metrics": metrics, "intervals": intervals}, results, curve


# ---------- report ----------

TABLE_ROWS = [
    ("mean_iou", "IoU of matched boxes (mean)"),
    ("mean_best_iou", "Best IoU per manual box (mean, misses = 0)"),
    ("precision", "Precision (PPV)"),
    ("recall", "Recall (sensitivity)"),
    ("f1", "F1-score"),
    ("ap50", "AP50"),
    ("ap75", "AP75"),
    ("ap", "mAP50-95"),
    ("ap_small", "AP50-95, small boxes"),
    ("ap_medium", "AP50-95, medium boxes"),
    ("ap_large", "AP50-95, large boxes"),
    ("fp_per_image", "False positives per image"),
    *[(fp_key(level), f"Sensitivity at {level:g} FP/image") for level in FP_LEVELS],
    ("froc_score", "FROC score (mean sensitivity, 1/8-8 FP/image)"),
    ("photo_any_found", "Photos with at least one wound found"),
    ("photo_all_found", "Photos with every wound found"),
    ("share_no_box", "Photos with no box at the threshold (fallback)"),
]


def formatted(value: float, interval: Optional[Dict[str, float]]) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    text = f"{value:.3f}"
    return f"{text} [{interval['low']:.3f}, {interval['high']:.3f}]" if interval else text


def report(arms: Dict[str, Path], output_dir: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.plots import CLASS_COLOURS, CLASS_DASHES, INK, MUTED

    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {arm: (json.loads((folder / "metrics.json").read_text(encoding="utf-8")),
                    pd.read_csv(folder / "froc.csv")) for arm, folder in arms.items()}
    label = {arm: ARM_LABELS.get(arm, arm) for arm in arms}

    # Overall table: one row per metric, one column per arm ("value [low, high]") plus the raw values
    rows = []
    for key, name in TABLE_ROWS:
        row = {"metric": name, "key": key}
        for arm, (m, _) in loaded.items():
            overall = m["overall"]
            row[label[arm]] = formatted(overall["metrics"].get(key, np.nan), overall["intervals"].get(key))
            row[f"{arm}_value"] = overall["metrics"].get(key, np.nan)
        rows.append(row)
    head = {"metric": "Operating threshold", "key": "threshold"}
    head.update({label[a]: f"{m['threshold']:.3f} ({m['chosen_config']})" for a, (m, _) in loaded.items()})
    count = {"metric": "Test photos / manual boxes / patients", "key": "n"}
    count.update({label[a]: f"{m['overall']['metrics']['n_photos']} / {m['overall']['metrics']['n_true']} / "
                            f"{m['overall']['metrics']['n_cases']}" for a, (m, _) in loaded.items()})
    pd.DataFrame([head, count, *rows]).to_csv(output_dir / "detection_metrics_table.csv", index=False)

    # Per wound type
    type_rows = []
    for arm, (m, _) in loaded.items():
        for wound, part in m["by_type"].items():
            v, ci = part["metrics"], part["intervals"]
            type_rows.append({"arm": label[arm], "wound_type": wound, "photos": v["n_photos"], "manual_boxes": v["n_true"],
                              "patients": v["n_cases"],
                              **{k: formatted(v.get(k, np.nan), ci.get(k)) for k in
                                 ("ap50", "ap75", "ap", "precision", "recall", "f1", "fp_per_image",
                                  "sensitivity_at_1fp", "mean_best_iou")}})
    pd.DataFrame(type_rows).to_csv(output_dir / "detection_by_type.csv", index=False)

    # FROC and precision-recall curves, with each arm's operating point
    fig, (froc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, (arm, (m, curve)) in enumerate(loaded.items()):
        colour, dash = CLASS_COLOURS[k % len(CLASS_COLOURS)], CLASS_DASHES[k % len(CLASS_DASHES)]
        overall = m["overall"]["metrics"]
        pts = curve[curve["fp_per_image"] > 0]
        froc_ax.plot(pts["fp_per_image"], pts["sensitivity"], color=colour, linestyle=dash, linewidth=2,
                     label=f"{label[arm]} (FROC score {overall['froc_score']:.2f})")
        froc_ax.plot(overall["fp_per_image"], overall["recall"], "o", color=colour, markersize=8,
                     markeredgecolor="white", markeredgewidth=1.5)
        pr = curve[curve["n_predicted"] > 0]
        pr_ax.plot(pr["sensitivity"], pr["precision"], color=colour, linestyle=dash, linewidth=2,
                   label=f"{label[arm]} (AP50 {overall['ap50']:.2f})")
        pr_ax.plot(overall["recall"], overall["precision"], "o", color=colour, markersize=8,
                   markeredgecolor="white", markeredgewidth=1.5)
    froc_ax.set_xscale("log", base=2)
    froc_ax.set_xlim(1 / 16, 16)
    froc_ax.set_xticks(FP_LEVELS)
    froc_ax.set_xticklabels([f"{x:g}" for x in FP_LEVELS])
    froc_ax.set_xlabel("False positives per image")
    froc_ax.set_ylabel("Sensitivity (manual boxes found, IoU ≥ 0.5)")
    froc_ax.set_title("FROC", loc="left", color=INK)
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_title("Precision–recall (IoU ≥ 0.5)", loc="left", color=INK)
    for ax in (froc_ax, pr_ax):
        ax.set_ylim(0, 1.02)
        ax.grid(color=MUTED, alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, 0.01, "Dots: operating point at each arm's validation-chosen threshold. Test photos.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_dir / "froc_pr.png", dpi=200)
    plt.close(fig)
    print(f"Wrote the comparison of {', '.join(arms)} to {output_dir}")
    return 0


# ---------- script ----------

def outside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(f"{resolved} is inside the repository; per-photo results stay outside it")
    return resolved


def evaluate_main(args: argparse.Namespace) -> int:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if "test" in args.split and not selection.get("gate_passed", False):
        raise SystemExit(f"{args.selection}: the arm did not pass its recall gate, so it never touches test (T19 §2.4)")
    out = outside_repo(args.output_dir)
    limits = json.loads(args.limits.read_text(encoding="utf-8"))
    photos = reference_photos(pd.read_csv(args.reference, dtype={PHOTO: str, GROUP_COLUMN: str}), args.split)
    if args.patient_cases is not None:
        photos = with_patients(photos, args.patient_cases)
    cands = candidates(load_detections(args.detections), photos, limits["max_box_share"])
    threshold, cap, bins = float(selection["threshold"]), int(limits["max_boxes_per_photo"]), limits["size_bins"]

    overall, results, curve = evaluate_photos(photos, cands, threshold, cap, bins, args.n_resamples, args.seed)
    by_type = {}
    for wound in sorted(photos["label"].unique()):
        part, _, _ = evaluate_photos(photos[photos["label"] == wound], cands, threshold, cap, bins,
                                     args.n_resamples, args.seed)
        by_type[wound] = part
    record = {"arm": selection.get("arm"), "chosen_config": selection.get("chosen_config"), "threshold": threshold,
              "detections": str(args.detections), "selection": str(args.selection), "split": args.split,
              "resampled_unit": "patient" if args.patient_cases is not None else "group",
              "n_resamples": args.n_resamples, "seed": args.seed, "fp_levels": list(FP_LEVELS),
              "overall": overall, "by_type": by_type}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    curve.to_csv(out / "froc.csv", index=False)
    results.to_csv(out / "photo_results.csv", index=False)
    m = overall["metrics"]
    print(f"{record['arm']} {'+'.join(args.split)}: AP50 {m['ap50']:.3f}, AP75 {m['ap75']:.3f}, mAP50-95 {m['ap']:.3f}, "
          f"P {m['precision']:.3f}, R {m['recall']:.3f}, F1 {m['f1']:.3f}, FP/img {m['fp_per_image']:.2f}, "
          f"sens@1FP {m[fp_key(1.0)]:.3f}. Wrote {out}")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full detection evaluation and arm comparison (T37).")
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--detections", type=Path, required=True)
    e.add_argument("--selection", type=Path, required=True, help="The arm's frozen selection.json.")
    e.add_argument("--split", nargs="+", default=["test"], choices=("train", "validation", "test"))
    e.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    e.add_argument("--reference", type=Path, default=DATA_DIR / "roi_auto/reference_boxes.csv")
    e.add_argument("--limits", type=Path, default=DATA_DIR / "roi_auto/reference_limits.json")
    e.add_argument("--patient-cases", type=Path, default=None,
                   help="splits/patient_cases.csv: resample patients (T25) instead of T23 groups.")
    e.add_argument("--n-resamples", type=int, default=2000)
    e.add_argument("--seed", type=int, default=42)
    r = sub.add_parser("report")
    r.add_argument("--arm", nargs="+", required=True, metavar="NAME=DIR",
                   help="Arm name and its evaluate output folder, e.g. 4Y=.../4Y; listed in plot order.")
    r.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        pairs = [a.split("=", 1) for a in args.arm]
        if any(len(p) != 2 for p in pairs):
            parser.error("--arm takes NAME=DIR")
        args.arms = {name: Path(folder) for name, folder in pairs}
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return evaluate_main(args) if args.command == "evaluate" else report(args.arms, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
