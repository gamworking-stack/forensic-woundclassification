#!/usr/bin/env python3
"""Verification of the per-box ROI run (T17), the T16 checks applied to S17-S24.

1. files:     8 of 8 scenarios hold crop-level and photo-level prediction, metric, report and figure
              files; the summary files exist.
2. test set:  crop files contain only test (validation) cases of the frozen split; the photo-level
              test set of every scenario equals the main run's test set.
3. selection: learning rate, SVM setting and ensemble members reproduced from validation data alone
              (T16 rules; per-box histories are checked on photo level), test predicted after the choice.
4. budget:    every per-box network shares the main run's training budget and seed policy.
5. numbers:   photo files, crop and photo metrics (with bootstrap intervals), evaluation_index.csv,
              box_summary_table.csv, box_vs_union.csv and box_vs_full.csv recomputed.
6. gradcam:   (if run) same example photos as the main run, one map per crop, the reported models.
7. cv:        (with --cv-root) no test case or test prediction, CV mode, folds respected, and the
              per-box CV table and comparisons recomputed.

Writes <box-root>/verification/verification_report.csv and exits 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.box_summary import CROP_METRICS, compare, summary_row  # noqa: E402
from src.evaluation.cv_box_summary import photo_frame_of, table_row  # noqa: E402
from src.evaluation.cv_summary import FOLD_METRICS  # noqa: E402
from src.evaluation.evaluate import BOX_PATH, BOX_SCENARIO_IDS, FAMILY_ORDER, SPLITS  # noqa: E402
from src.evaluation.gradcam import FAMILIES as GRADCAM_FAMILIES, checkpoint_folder  # noqa: E402
from src.evaluation.metrics import case_bootstrap, compute_metrics, load_predictions, summary_metrics  # noqa: E402
from src.evaluation.summarize import load_pairs  # noqa: E402
from src.evaluation.verify import (  # noqa: E402
    BUDGET_KEYS, SCENARIO_FILES, SINGLE_FAMILIES, TOLERANCE, Report,
    check_ensemble, check_lr_selection, check_svm_selection, close, read_json,
)
from src.training.photo import MANIFEST_PHOTO_COLUMN, PHOTO_COLUMN  # noqa: E402


PHOTO_FILES = tuple(n for n in SCENARIO_FILES if n != "run_config.json")
SUMMARY_FILES = ("evaluation_index.csv", "summary/box_summary_table.csv", "summary/box_vs_union.csv",
                 "summary/box_vs_full.csv", "summary/box_vs_union_macro_f1.png")
PAIRED_COLUMNS = ["reference", "box", "difference"]
PAIRED_BOOT_COLUMNS = ["ci_low", "ci_high", "p_value", "p_holm", "resamples"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verification of the per-box ROI run (T17).")
    parser.add_argument("--box-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, default=Path("data/splits/case_split_assignments.csv"))
    parser.add_argument("--box-manifest", type=Path, default=Path("data/roi_box/roi_manifest.csv"))
    parser.add_argument("--cv-root", type=Path, default=None, help="Per-box CV run; omitted: CV not checked.")
    parser.add_argument("--cv-main-root", type=Path, default=None, help="T12b CV run (needed with --cv-root).")
    parser.add_argument("--cv-folds", type=Path, default=Path("data/splits/cv_folds.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000, help="0 skips the interval checks.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if (args.cv_root is None) != (args.cv_main_root is None):
        parser.error("--cv-root and --cv-main-root go together")
    return args


def box_scenarios():
    for family in FAMILY_ORDER:
        yield BOX_SCENARIO_IDS[(BOX_PATH, family)], family


def scored_photo_frame(crop_path: Path) -> pd.DataFrame:
    return photo_frame_of(crop_path)


def as_predictions(frame: pd.DataFrame, split: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{split}.csv"
        frame.to_csv(path, index=False)
        return load_predictions(path, split)


# 1. Files -----------------------------------------------------------------------------------------

def check_files(report: Report, root: Path) -> None:
    for sid, family in box_scenarios():
        folder = root / BOX_PATH / family
        missing = [n for n in SCENARIO_FILES if not (folder / n).is_file()]
        missing += [f"photo/{n}" for n in PHOTO_FILES if not (folder / "photo" / n).is_file()]
        bad_png = [str(p.relative_to(folder)) for p in list(folder.glob("*.png")) + list((folder / "photo").glob("*.png"))
                   if p.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"]
        report.add("files", f"{sid} box/{family}", not missing and not bad_png,
                   f"missing {missing}, not PNG {bad_png}" if missing or bad_png
                   else f"{len(SCENARIO_FILES) + len(PHOTO_FILES)} files incl. 8 figures (crop and photo level)")
    for name in SUMMARY_FILES:
        report.add("files", name, (root / name).is_file())


# 2. Test set --------------------------------------------------------------------------------------

def check_test_set(report: Report, root: Path, main_root: Path, assignments: Dict[str, str]) -> None:
    reference = load_pairs(main_root)[FAMILY_ORDER[0]][1].signature()
    differ = []
    for sid, family in box_scenarios():
        folder = root / BOX_PATH / family

        def run(folder=folder, sid=sid, family=family) -> None:
            for split in SPLITS:
                p = load_predictions(folder / f"{split}_predictions.csv", split)
                wrong = sorted({c for c in p.cases if assignments.get(c) != split})
                no_photo = PHOTO_COLUMN not in p.frame.columns or p.frame[PHOTO_COLUMN].isna().any()
                report.add("test set", f"{sid} {split} crops", not wrong and not no_photo,
                           f"{len(set(p.cases))} cases, {len(p.cases)} crops"
                           + (f"; not {split}: {wrong[:5]}" if wrong else "") + ("; photo column missing" if no_photo else ""))
            photo = load_predictions(folder / "photo" / "test_predictions.csv", "test")
            if photo.signature() != reference:
                differ.append(family)
        report.guard("test set", f"{sid} box/{family}", run)
    report.add("test set", "photo-level test set = main run's test set (all 8)", not differ,
               f"differ: {differ}" if differ else "same cases and images per class as S01-S16")


# 3. Selection -------------------------------------------------------------------------------------

def check_selection(report: Report, root: Path) -> Dict[str, Path]:
    chosen: Dict[str, Path] = {}
    for sid, family in box_scenarios():
        folder, item = root / BOX_PATH / family, f"{sid} box/{family}"

        def run(folder=folder, item=item, family=family) -> None:
            if family in SINGLE_FAMILIES:
                chosen[family] = check_lr_selection(report, folder, item)
            elif family == "cnn_svm":
                check_svm_selection(report, folder, item)
            else:
                chosen[family] = check_ensemble(report, folder, item)
        report.guard("selection", item, run)
    units = Counter(read_json(p).get("selection_unit") for p in root.glob(f"{BOX_PATH}/*/*/run_config.json"))
    report.add("selection", "trained networks select on photo level", set(units) == {"photo"}, str(dict(units)))
    return chosen


# 4. Budget ----------------------------------------------------------------------------------------

def check_budget(report: Report, root: Path, main_root: Path) -> None:
    box = {str(p.parent.relative_to(root)): read_json(p) for p in sorted(root.glob(f"{BOX_PATH}/*/*/run_config.json"))}
    main = [read_json(p) for p in sorted(main_root.glob("*/*/lr_*/run_config.json"))[:1]]
    for key in BUDGET_KEYS:
        values = Counter(json.dumps(c.get(key)) for c in box.values())
        same_as_main = not main or json.dumps(main[0].get(key)) in values
        report.add("budget", key, len(values) == 1 and same_as_main,
                   f"{dict(values)} over {len(box)} networks; main {json.dumps(main[0].get(key)) if main else '?'}")
    commits = Counter((c.get("git_commit", "?")[:7], c.get("git_dirty")) for c in box.values())
    report.add("budget", "code version (commit, dirty)", None, str(dict(commits)))


# 5. Numbers ---------------------------------------------------------------------------------------

def check_numbers(report: Report, root: Path, main_root: Path, n_boot: int, seed: int) -> None:
    fresh_photo: Dict[str, object] = {}
    rows = []
    for sid, family in box_scenarios():
        folder = root / BOX_PATH / family

        def run(folder=folder, sid=sid, family=family) -> None:
            saved_metrics = {}
            for level, base in (("crop", folder), ("photo", folder / "photo")):
                for split in SPLITS:
                    if level == "photo":
                        frame = scored_photo_frame(folder / f"{split}_predictions.csv")
                        saved_frame = pd.read_csv(base / f"{split}_predictions.csv", dtype=str)
                        prob = [c for c in frame.columns if c.startswith("prob_")]
                        same = list(frame["relative_path"]) == list(saved_frame["relative_path"]) and \
                            list(frame["predicted_class"]) == list(saved_frame["predicted_class"]) and \
                            np.allclose(frame[prob].to_numpy(), saved_frame[prob].astype(float).to_numpy(), atol=1e-12)
                        report.add("numbers", f"{sid} photo/{split} = mean of crops", same)
                        p = as_predictions(frame, split)
                    else:
                        p = load_predictions(base / f"{split}_predictions.csv", split)
                    fresh = compute_metrics(p)
                    saved = read_json(base / f"{split}_metrics.json")
                    off = [k for k, v in fresh.items() if not close(v, saved[k])]
                    if split == "test" and n_boot:
                        boot = saved["bootstrap"]
                        ci = case_bootstrap(p, boot["resamples"], boot["seed"])
                        off += [f"{m} CI" for m, v in ci.items() if not (
                            close(v["low"], saved["confidence_intervals"][m]["low"])
                            and close(v["high"], saved["confidence_intervals"][m]["high"]))]
                    if split == "test":
                        saved_metrics[level] = saved
                        if level == "photo":
                            fresh_photo[family] = p
                    report.add("numbers", f"{sid} {level} {split} metrics", not off,
                               f"macro-F1 {fresh['macro_f1']:.4f}" + (f"; differ: {off}" if off else ""))
            rows.append(summary_row(family, saved_metrics["photo"], saved_metrics["crop"]))
        report.guard("numbers", f"{sid} box/{family}", run)

    def tables() -> None:
        index = pd.read_csv(root / "evaluation_index.csv")
        report.add("numbers", "evaluation_index.csv", len(index) == 8 and (index["status"] == "evaluated").all()
                   and index["scenario_id"].tolist() == [sid for sid, _ in box_scenarios()],
                   f"{len(index)} rows")
        saved = pd.read_csv(root / "summary" / "box_summary_table.csv")
        fresh = pd.DataFrame(rows)
        numeric = [c for c in fresh.columns if c not in ("scenario_id", "family")]
        off = [c for c in numeric if not np.allclose(saved[c], fresh[c], rtol=0, atol=TOLERANCE)]
        report.add("numbers", "box_summary_table.csv", saved["family"].tolist() == fresh["family"].tolist() and not off,
                   f"differ: {off}" if off else f"{len(saved)} rows, all columns")
        pairs = load_pairs(main_root)
        for name, k in (("box_vs_union.csv", 1), ("box_vs_full.csv", 0)):
            compare_saved(report, name, pd.read_csv(root / "summary" / name),
                          {f: pairs[f][k] for f in FAMILY_ORDER}, fresh_photo, n_boot, seed)
    report.guard("numbers", "tables", tables)


def compare_saved(report: Report, item: str, saved: pd.DataFrame, reference, box, n_boot: int, seed: int) -> None:
    fresh = compare(reference, box, n_boot or 1, seed)
    columns = PAIRED_COLUMNS + (PAIRED_BOOT_COLUMNS if n_boot else [])
    same_keys = saved[["family", "metric"]].equals(fresh[["family", "metric"]])
    off = [c for c in columns if not np.allclose(saved[c], fresh[c], rtol=0, atol=TOLERANCE, equal_nan=True)]
    report.add("numbers", item, same_keys and not off,
               (f"differ: {off}" if off else f"{len(saved)} rows, columns {columns}")
               + ("" if n_boot else "; intervals not rechecked (--bootstrap 0)"))


# 6. Grad-CAM --------------------------------------------------------------------------------------

def check_gradcam(report: Report, root: Path, main_root: Path, box_manifest: Path, chosen: Dict[str, Path]) -> None:
    out = root / "gradcam"
    if not (out / "gradcam_results.csv").is_file():
        report.add("gradcam", "per-box Grad-CAM", None, "not run")
        return
    examples = pd.read_csv(out / "examples.csv", dtype=str, keep_default_na=False)
    main_examples = pd.read_csv(main_root / "gradcam" / "examples.csv", dtype=str, keep_default_na=False)
    report.add("gradcam", "same example photos as the main run",
               examples[MANIFEST_PHOTO_COLUMN].tolist() == main_examples[MANIFEST_PHOTO_COLUMN].tolist(),
               f"{len(examples)} photos")
    crops = pd.read_csv(box_manifest, dtype=str, keep_default_na=False)
    n_crops = int(crops[MANIFEST_PHOTO_COLUMN].isin(examples[MANIFEST_PHOTO_COLUMN]).sum())
    results = pd.read_csv(out / "gradcam_results.csv", dtype={"case": str})
    expected = len(GRADCAM_FAMILIES) * n_crops
    report.add("gradcam", "one map per family and crop", len(results) == expected and
               not results.duplicated(["family", "image"]).any(), f"{len(results)} of {expected} ({n_crops} crops)")
    wrong = [f for f, rows in results.groupby("family")
             if f not in chosen or not rows["checkpoint"].map(
                 lambda c, m=chosen[f]: Path(c).parent.parts[-3:] == m.parts[-3:]).all()]
    report.add("gradcam", "maps use the reported models", not wrong, f"differ: {wrong}" if wrong else "")
    report.add("gradcam", "predictions equal the prediction files", None,
               f"crops {int(results['matches_prediction_file'].sum())} of {len(results)}; photos "
               f"{int(results.drop_duplicates(['family', 'photo'])['matches_photo_prediction_file'].sum())} of "
               f"{len(results.drop_duplicates(['family', 'photo']))}")
    figures = [f"{f}_set{k}.png" for f in GRADCAM_FAMILIES for k in sorted(examples["set"].unique())]
    missing = [f for f in figures if not (out / f).is_file() or (out / f).read_bytes()[:4] != b"\x89PNG"]
    report.add("gradcam", "figures", not missing, f"missing {missing}" if missing else f"{len(figures)} figures")


# 7. Cross-validation ------------------------------------------------------------------------------

def check_cv(report: Report, cv_root: Path, cv_main_root: Path, cv_folds: Path, folds: int,
             assignments: Dict[str, str], n_boot: int, seed: int) -> None:
    fold_of = dict(pd.read_csv(cv_folds, dtype=str).itertuples(index=False, name=None))
    test_files = sorted(str(p.relative_to(cv_root)) for p in cv_root.rglob("test_predictions.csv"))
    report.add("cv", "no CV run predicted test", not test_files, f"found {test_files[:5]}" if test_files else "")
    not_cv = sorted(str(p.parent.relative_to(cv_root)) for p in cv_root.rglob("run_config.json")
                    if read_json(p).get("family") != "ensemble" and not read_json(p).get("cv_folds"))
    report.add("cv", "every CV run ran in CV mode (test rows dropped)", not not_cv, f"not CV: {not_cv[:5]}")

    def run() -> None:
        saved = pd.read_csv(cv_root / "summary" / "cv_box_table.csv")
        rows, box, off = [], {}, []
        for sid, family in box_scenarios():
            frames, per_fold = [], []
            for k in range(1, folds + 1):
                path = cv_root / BOX_PATH / family / f"fold_{k}" / "validation_predictions.csv"
                crops = load_predictions(path, "validation")
                outside = sorted({c for c in crops.cases if fold_of.get(c) != str(k) or assignments.get(c) == "test"})
                if outside:
                    off.append(f"{sid} fold {k}: cases outside the fold {outside[:3]}")
                frame = scored_photo_frame(path)
                p = as_predictions(frame, "validation")
                per_fold.append(summary_metrics(p.y_true, p.y_pred, p.probs))
                frames.append(frame)
            oof = as_predictions(pd.concat(frames, ignore_index=True), "validation")
            rows.append(table_row(family, folds, per_fold, oof))
            box[family] = oof
        fresh = pd.DataFrame(rows)
        numeric = [f"{m}_{s}" for m in FOLD_METRICS for s in ("mean", "sd", "oof")] + ["oof_photos", "oof_cases"]
        off += [c for c in numeric if not np.allclose(saved[c], fresh[c], rtol=0, atol=TOLERANCE)]
        report.add("cv", "cv_box_table.csv (fold mean, SD, out of fold, photo level)", not off,
                   f"differ: {off[:5]}" if off else f"8 scenarios x {folds} folds")
        for name, path in (("cv_box_vs_union.csv", "roi"), ("cv_box_vs_full.csv", "full")):
            reference = {f: load_predictions(cv_main_root / path / f / "oof_predictions.csv", "validation")
                         for f in FAMILY_ORDER}
            compare_saved(report, name, pd.read_csv(cv_root / "summary" / name), reference, box, n_boot, seed)
    report.guard("cv", "recompute", run)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = args.box_root
    assignments = dict(pd.read_csv(args.split_assignments, dtype=str).itertuples(index=False, name=None))
    report = Report()

    check_files(report, root)
    check_test_set(report, root, args.main_root, assignments)
    chosen = check_selection(report, root)
    report.guard("budget", "configs", lambda: check_budget(report, root, args.main_root))
    check_numbers(report, root, args.main_root, args.bootstrap, args.seed)
    report.guard("gradcam", "outputs", lambda: check_gradcam(report, root, args.main_root, args.box_manifest, chosen))
    if args.cv_root is not None:
        check_cv(report, args.cv_root, args.cv_main_root, args.cv_folds, args.folds, assignments,
                 args.bootstrap, args.seed)

    out = root / "verification"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(report.rows, columns=["check", "item", "status", "detail"])
    table.to_csv(out / "verification_report.csv", index=False)
    for check, rows in table.groupby("check", sort=False):
        counts = rows["status"].value_counts().to_dict()
        print(f"{check:<10} " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), flush=True)
    for row in table[table["status"] != "PASS"].to_dict("records"):
        print(f"  {row['status']} [{row['check']}] {row['item']}: {row['detail']}")
    print(f"{'FAILED' if report.failed else 'All checks passed'}; report written to {out / 'verification_report.csv'}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
