#!/usr/bin/env python3
"""Verification of the auto-box ROI experiment (T19 §3.8), in the style of T16.

1. reference: every unpadded manual box, padded as T03 pads it, reproduces its per-box manifest crop.
2. selection: per arm,
   - the detections files match the hashes in ``selection.json``;
   - no test photo was in them;
   - the configuration, threshold and gate verdict are reproduced from them.
3. order:     test detections exist only for an arm that passed its gate, and were written after its
              ``selection.json``; a failed arm has no test detections, crops or primary predictions.
4. crossfit:  (4F) every training photo is boxed exactly once, by the detector trained without its fold;
              the cross-fitted runs use the chosen learning rate and its best epoch count.
5. crops:     the auto crops are rebuilt from the frozen detections and threshold and match the
              manifests; every arm's test set is the 157 manual test photos.
6. primary:   every primary prediction file covers exactly its manifest's test rows.
7. numbers:   the observed differences of ``primary_noninferiority.csv`` are recomputed.
8. budget:    retrained networks share the main run's training budget.
9. weights:   (4P) the original checkpoints have the SHA-256 recorded by the research (T19 §2.2).

Writes ``<experiments-dir>/auto_summary/verification/verification_report.csv``; exits 1 on any FAIL.
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
from src.data.create_auto_roi_dataset import build_rows, final_boxes_by_photo  # noqa: E402
from src.detection.boxes import COORDINATES, padded_crop_box  # noqa: E402
from src.detection.common import PHOTO, photo_table  # noqa: E402
from src.detection.select import file_sha256, select  # noqa: E402
from src.evaluation.auto_vs_manual import DESIGNS, PRIMARY_METRIC, predictions  # noqa: E402
from src.evaluation.evaluate import FAMILY_ORDER  # noqa: E402
from src.evaluation.metrics import summary_metrics  # noqa: E402
from src.evaluation.verify import BUDGET_KEYS, Report, read_json  # noqa: E402
from src.training.data import GROUP_COLUMN  # noqa: E402


ARMS = ("4Z", "4P", "4F")
HANTERDSITH_SHA256_PREFIX = {"v8n": "29be848ee1bc932b", "v8m": "b7988689dff882b3", "v8x": "dfa0d8de92f9bf8b"}
NUMBER_TOLERANCE = 1e-9


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verification of the auto-box ROI experiment (T19).")
    parser.add_argument("--experiments-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/roi_auto"))
    parser.add_argument("--box-manifest", type=Path, default=Path("data/roi_box/roi_manifest.csv"))
    parser.add_argument("--cv-folds", type=Path, default=Path("data/splits/cv_folds.csv"))
    return parser.parse_args(argv)


def check_reference(report: Report, manifest_root: Path, box_manifest: Path) -> None:
    reference = pd.read_csv(manifest_root / "reference_boxes.csv", dtype={PHOTO: str})
    crops = pd.read_csv(box_manifest, dtype=str).set_index([PHOTO, "crop_index"])
    # Pad every reference box and count those that differ from their per-box crop
    bad = 0
    for row in reference.to_dict("records"):
        crop = crops.loc[(row[PHOTO], str(row["box_index"]))]
        padded = padded_crop_box([[row[c] for c in COORDINATES]], row["source_width"], row["source_height"])
        bad += padded != tuple(int(crop[c]) for c in COORDINATES)
    report.add("reference", "padded boxes = per-box crops", bad == 0, f"{bad} of {len(reference)} differ")
    # The box counts per split must be the known 1,177 / 277 / 213
    limits = read_json(manifest_root / "reference_limits.json")
    report.add("reference", "boxes per split", limits["boxes_by_split"] == {"test": 213, "train": 1177, "validation": 277},
               json.dumps(limits["boxes_by_split"]))


def check_selection(report: Report, arm: str, detection: Path, reference: pd.DataFrame, limits: Dict) -> Dict:
    selection = read_json(detection / "selection.json")
    # Each selection file must still have the hash recorded at selection time
    dev = Path(selection["detections_dir"])
    for config, digest in selection["detections_sha256"].items():
        report.add("selection", f"{arm} {config} hash", file_sha256(dev / f"{config}.csv") == digest)
    # No test photo may have been in the selection data
    dropped = sum(selection["test_detections_dropped"].values())
    report.add("selection", f"{arm} no test photo in selection data", dropped == 0, f"{dropped} test rows dropped")
    # Rerun the selection from the same files: same choice, threshold and gate?
    again = select(arm, [c["config"] for c in selection["configs"]], dev, reference, limits)
    same = (again["chosen_config"] == selection["chosen_config"] and again["gate_passed"] == selection["gate_passed"]
            and np.isclose(again["threshold"], selection["threshold"], equal_nan=True))
    report.add("selection", f"{arm} choice reproduced", bool(same),
               f"{again['chosen_config']} @ {again['threshold']:.6g}, gate {again['gate_passed']}")
    return selection


def check_order(report: Report, arm: str, root: Path, selection: Dict, manifest_root: Path) -> None:
    # Find the arm's test detections and when selection.json was written
    test_dir = root / "detection" / "test"
    test_files = sorted(test_dir.glob("*.csv")) if test_dir.is_dir() else []
    selection_time = (root / "detection" / "selection.json").stat().st_mtime
    # Passed: only the chosen configuration was run on test, and only after the selection
    if selection["gate_passed"]:
        report.add("order", f"{arm} test detections of the chosen config", [p.stem for p in test_files] == [selection["chosen_config"]],
                   str([p.name for p in test_files]))
        late = all(p.stat().st_mtime >= selection_time for p in test_files)
        report.add("order", f"{arm} test detections written after selection.json", late)
    else:
        # Failed: no test detections, crops or primary predictions may exist
        leftovers = [str(p) for p in (*test_files, manifest_root / arm, root / "primary") if Path(p).exists()]
        report.add("order", f"{arm} failed gate: nothing downstream", not leftovers, str(leftovers))


def check_crossfit(report: Report, root: Path, selection: Dict, box_manifest: Path, cv_folds: Path) -> List[Path]:
    # Fold of each case, the training photos, and the chosen run's best epoch
    folds = pd.read_csv(cv_folds, dtype=str).set_index(GROUP_COLUMN)["fold"].astype(int)
    train = photo_table(box_manifest, ("train",)).set_index(PHOTO)
    chosen = root / "detector" / selection["chosen_config"]
    history = pd.read_csv(chosen / "history.csv")
    best_epoch = int(history.loc[history["validation_ap50"].idxmax(), "epoch"])
    files, seen = [], Counter()
    for k in sorted(set(folds)):
        # Each fold run must use the chosen learning rate and the best epoch count
        run = root / "detector" / "crossfit" / f"fold{k}"
        config = read_json(run / "run_config.json")
        report.add("crossfit", f"fold {k} recipe", config["lr"] == read_json(chosen / "run_config.json")["lr"]
                   and int(config["epochs"]) == best_epoch, f"lr {config['lr']}, epochs {config['epochs']} (best {best_epoch})")
        # Each fold run may box only the photos of its own (held-out) fold
        detections = pd.read_csv(run / f"train_fold{k}.csv", dtype={PHOTO: str})
        photos = set(detections[PHOTO])
        wrong = [p for p in photos if folds[train.loc[p, GROUP_COLUMN]] != k]
        report.add("crossfit", f"fold {k} boxes only its own photos", not wrong, f"{len(wrong)} photos of other folds")
        seen.update(set(train.index[train[GROUP_COLUMN].map(folds) == k]))
        files.append(run / f"train_fold{k}.csv")
    # Together the folds cover every training photo exactly once
    report.add("crossfit", "every training photo in exactly one fold", set(seen) == set(train.index) and max(seen.values()) == 1,
               f"{len(seen)} of {len(train)} photos")
    return files


def check_crops(report: Report, arm: str, root: Path, selection: Dict, limits: Dict, manifest_root: Path,
                box_manifest: Path, detection_files: List[Path]) -> None:
    # Rebuild the crops from the frozen detections with the same code that made them
    detections = pd.concat([pd.read_csv(p, dtype={PHOTO: str}) for p in detection_files], ignore_index=True)
    photos = photo_table(box_manifest)
    rows = build_rows(photos, final_boxes_by_photo(photos, detections, float(selection["threshold"]), limits), arm)
    for design in DESIGNS:
        # Compare with the written manifest: same crops, same coordinates, same test photos
        manifest = pd.read_csv(manifest_root / arm / design / "roi_manifest.csv", dtype={PHOTO: str})
        rebuilt = pd.DataFrame(rows[design])
        same = (len(manifest) == len(rebuilt) and (manifest["roi_relative_path"].to_numpy() == rebuilt["roi_relative_path"].to_numpy()).all()
                and np.array_equal(manifest[COORDINATES].to_numpy(int), rebuilt[COORDINATES].to_numpy(int)))
        report.add("crops", f"{arm} {design} rebuilt from frozen detections", bool(same), f"{len(manifest)} crops")
        test_photos = set(manifest.loc[manifest["split"] == "test", PHOTO])
        report.add("crops", f"{arm} {design} test photos = manual test photos",
                   test_photos == set(photos.loc[photos["split"] == "test", PHOTO]), f"{len(test_photos)} photos")


def check_primary(report: Report, arm: str, root: Path, manifest_root: Path) -> None:
    for design, path in DESIGNS.items():
        # Each family's primary predictions must cover exactly the manifest's test crops
        manifest = pd.read_csv(manifest_root / arm / design / "roi_manifest.csv", dtype=str)
        expected = set(manifest.loc[manifest["split"] == "test", "roi_relative_path"])
        for family in FAMILY_ORDER:
            frame = pd.read_csv(root / "primary" / path / family / "test_predictions.csv", dtype=str)
            report.add("primary", f"{arm} {design} {family} rows", set(frame["relative_path"]) == expected,
                       f"{len(frame)} rows, {len(expected)} test crops")


def check_numbers(report: Report, exp: Path, arms: List[str]) -> None:
    # Recompute each reported pooled difference from the prediction files
    table = pd.read_csv(exp / "auto_summary" / "primary_noninferiority.csv")
    cache = exp / "auto_summary" / "verification" / "photo_level"
    for row in table.to_dict("records"):
        if row["arm"] not in arms:
            continue
        path = DESIGNS[row["design"]]
        manual_root = exp / ("main" if row["design"] == "union" else "roi_box") / path
        diffs = []
        for family in FAMILY_ORDER:
            m = predictions(manual_root / family, row["design"], cache / "manual" / row["design"] / family)
            a = predictions(exp / f"auto_{row['arm']}" / "primary" / path / family, row["design"],
                            cache / row["arm"] / row["design"] / family)
            diffs.append(summary_metrics(a.y_true, a.y_pred, a.probs)[PRIMARY_METRIC]
                         - summary_metrics(m.y_true, m.y_pred, m.probs)[PRIMARY_METRIC])
        report.add("numbers", f"{row['arm']} {row['design']} pooled difference",
                   abs(float(np.mean(diffs)) - row["difference"]) <= NUMBER_TOLERANCE,
                   f"recomputed {np.mean(diffs):+.6f}, reported {row['difference']:+.6f}")


def check_budget(report: Report, exp: Path, arm: str) -> None:
    # Every retrained network's settings must equal the main run's budget
    retrained = [read_json(p) for p in sorted(exp.glob(f"auto_{arm}_*/*/*/lr_*/run_config.json"))
                 + sorted(exp.glob(f"auto_{arm}_*/*/*/member_*/run_config.json"))]
    if not retrained:
        report.add("budget", f"{arm} retrained networks", None, "none found (retraining not run yet)")
        return
    main = read_json(sorted(exp.glob("main/*/*/lr_*/run_config.json"))[0])
    for key in BUDGET_KEYS:
        if key == "class_weights":
            continue  # computed from each design's own crop counts
        values = Counter(json.dumps(c.get(key)) for c in retrained)
        report.add("budget", f"{arm} {key}", set(values) == {json.dumps(main.get(key))},
                   f"{dict(values)} over {len(retrained)} networks; main {json.dumps(main.get(key))}")


def check_weights(report: Report, root: Path) -> None:
    # The original 4P checkpoints must be the files the research identified
    record = read_json(root / "weights" / "weights.json")
    for size, prefix in HANTERDSITH_SHA256_PREFIX.items():
        digest = record["checkpoints"][size]["original_sha256"]
        report.add("weights", f"4P {size} original checkpoint", digest.startswith(prefix), digest[:16])


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    exp = args.experiments_dir
    report = Report()
    # Reference boxes first; then the inputs every arm check needs
    report.guard("reference", "files", lambda: check_reference(report, args.manifest_root, args.box_manifest))
    reference = pd.read_csv(args.manifest_root / "reference_boxes.csv", dtype={PHOTO: str, GROUP_COLUMN: str})
    limits = read_json(args.manifest_root / "reference_limits.json")

    for arm in args.arms:
        # Per arm: selection, order, weights (4P); then, if it passed, cross-fitting (4F), crops, primary files, budget
        root = exp / f"auto_{arm}"
        selection = {}

        def selection_step():
            selection.update(check_selection(report, arm, root / "detection", reference, limits))
        report.guard("selection", arm, selection_step)
        if not selection:
            continue
        report.guard("order", arm, lambda: check_order(report, arm, root, selection, args.manifest_root))
        if arm == "4P":
            report.guard("weights", arm, lambda: check_weights(report, root))
        if not selection["gate_passed"]:
            continue

        def crops_step():
            dev = Path(selection["detections_dir"]) / f"{selection['chosen_config']}.csv"
            test = root / "detection" / "test" / f"{selection['chosen_config']}.csv"
            if arm == "4F":  # validation from the main detector, training photos cross-fitted
                files = check_crossfit(report, root, selection, args.box_manifest, args.cv_folds) + [dev, test]
            else:
                files = [dev, test]
            check_crops(report, arm, root, selection, limits, args.manifest_root, args.box_manifest, files)
        report.guard("crops", arm, crops_step)
        report.guard("primary", arm, lambda: check_primary(report, arm, root, args.manifest_root))
        report.guard("budget", arm, lambda: check_budget(report, exp, arm))
    # Finally the reported numbers
    report.guard("numbers", "primary", lambda: check_numbers(report, exp, args.arms))

    # Write the report, print counts per check and every non-PASS row; exit 1 on any FAIL
    out = exp / "auto_summary" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(report.rows, columns=["check", "item", "status", "detail"])
    table.to_csv(out / "verification_report.csv", index=False)
    for check, group in table.groupby("check", sort=False):
        counts = group["status"].value_counts().to_dict()
        print(f"{check:<10} " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), flush=True)
    for row in table[table["status"] != "PASS"].to_dict("records"):
        print(f"  {row['status']} [{row['check']}] {row['item']}: {row['detail']}")
    print(f"{'FAILED' if report.failed else 'All checks passed'}; report written to {out / 'verification_report.csv'}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
