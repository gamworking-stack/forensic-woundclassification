#!/usr/bin/env python3
"""Verification of the per-crop ROI run (T18), the T16 checks applied to S25-S32.

1. files:     8 of 8 scenarios hold prediction, metric, report and figure files; the summary exists.
2. test set:  prediction files are crop level (no photo column), contain only the frozen split's
              cases, and every scenario is tested on the same crops: the manifest's test crops.
3. selection: learning rate, SVM setting and ensemble members reproduced from validation data alone
              (T16 rules, on crop level), test predicted after the choice; networks select per crop.
4. budget:    every network shares the main run's training budget and seed policy, without weights.
5. numbers:   crop metrics (with bootstrap intervals), evaluation_index.csv and crop_summary_table.csv
              recomputed.

Writes <crop-root>/verification/verification_report.csv and exits 1 on any FAIL.
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
from src.evaluation.crop_summary import summary_row  # noqa: E402
from src.evaluation.evaluate import CROP_PATH, CROP_SCENARIO_IDS, FAMILY_ORDER, SPLITS  # noqa: E402
from src.evaluation.metrics import case_bootstrap, compute_metrics, load_predictions  # noqa: E402
from src.evaluation.verify import (  # noqa: E402
    BUDGET_KEYS, SCENARIO_FILES, SINGLE_FAMILIES, TOLERANCE, Report,
    check_ensemble, check_lr_selection, check_svm_selection, close, read_json,
)
from src.training.photo import PHOTO_COLUMN  # noqa: E402


SUMMARY_FILES = ("evaluation_index.csv", "summary/crop_summary_table.csv")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verification of the per-crop ROI run (T18).")
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True, help="Main run, for the training budget.")
    parser.add_argument("--split-assignments", type=Path, default=Path("data/splits/case_split_assignments.csv"))
    parser.add_argument("--crop-manifest", type=Path, default=Path("data/roi_box/roi_manifest.csv"))
    parser.add_argument("--bootstrap", type=int, default=2000, help="0 skips the interval checks.")
    return parser.parse_args(argv)


def crop_scenarios():
    for family in FAMILY_ORDER:
        yield CROP_SCENARIO_IDS[(CROP_PATH, family)], family


# 1. Files -----------------------------------------------------------------------------------------

def check_files(report: Report, root: Path) -> None:
    for sid, family in crop_scenarios():
        folder = root / CROP_PATH / family
        missing = [n for n in SCENARIO_FILES if not (folder / n).is_file()]
        bad_png = [p.name for p in folder.glob("*.png") if p.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"]
        report.add("files", f"{sid} crop/{family}", not missing and not bad_png,
                   f"missing {missing}, not PNG {bad_png}" if missing or bad_png
                   else f"{len(SCENARIO_FILES)} files incl. 4 figures")
    for name in SUMMARY_FILES:
        report.add("files", name, (root / name).is_file())


# 2. Test set --------------------------------------------------------------------------------------

def check_test_set(report: Report, root: Path, assignments: Dict[str, str], manifest: Path) -> None:
    rows = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    expected = sorted(rows.loc[rows["split"].str.strip().str.lower() == "test", "roi_relative_path"])
    differ = []
    for sid, family in crop_scenarios():
        folder = root / CROP_PATH / family

        def run(folder=folder, sid=sid, family=family) -> None:
            for split in SPLITS:
                p = load_predictions(folder / f"{split}_predictions.csv", split)
                wrong = sorted({c for c in p.cases if assignments.get(c) != split})
                photo = PHOTO_COLUMN in p.frame.columns
                report.add("test set", f"{sid} {split} crops", not wrong and not photo,
                           f"{len(set(p.cases))} cases, {len(p.cases)} crops"
                           + (f"; not {split}: {wrong[:5]}" if wrong else "")
                           + ("; photo column present (photo-level scoring)" if photo else ""))
                if split == "test" and sorted(p.frame["relative_path"]) != expected:
                    differ.append(family)
        report.guard("test set", f"{sid} crop/{family}", run)
    report.add("test set", "test crops = the manifest's test crops (all 8)", not differ,
               f"differ: {differ}" if differ else f"{len(expected)} crops in every scenario")


# 3. Selection -------------------------------------------------------------------------------------

def check_selection(report: Report, root: Path) -> None:
    for sid, family in crop_scenarios():
        folder, item = root / CROP_PATH / family, f"{sid} crop/{family}"

        def run(folder=folder, item=item, family=family) -> None:
            if family in SINGLE_FAMILIES:
                check_lr_selection(report, folder, item)
            elif family == "cnn_svm":
                check_svm_selection(report, folder, item)
            else:
                check_ensemble(report, folder, item)
        report.guard("selection", item, run)
    configs = [read_json(p) for p in root.glob(f"{CROP_PATH}/*/*/run_config.json")]
    units = Counter((c.get("selection_unit"), c.get("loss_weights")) for c in configs)
    report.add("selection", "trained networks select per crop, without loss weights",
               bool(configs) and set(units) == {("crop", "none")}, str(dict(units)))


# 4. Budget ----------------------------------------------------------------------------------------

def check_budget(report: Report, root: Path, main_root: Path) -> None:
    crop = [read_json(p) for p in sorted(root.glob(f"{CROP_PATH}/*/*/run_config.json"))]
    main = [read_json(p) for p in sorted(main_root.glob("*/*/lr_*/run_config.json"))[:1]]
    for key in BUDGET_KEYS:
        if key == "class_weights":  # computed from crop counts here, so they differ from the main run
            values = Counter(json.dumps(c.get(key) is not None) for c in crop)
            report.add("budget", key, set(values) == {"true"}, f"class weights set in {dict(values)} networks")
            continue
        values = Counter(json.dumps(c.get(key)) for c in crop)
        same_as_main = not main or json.dumps(main[0].get(key)) in values
        report.add("budget", key, len(values) == 1 and same_as_main,
                   f"{dict(values)} over {len(crop)} networks; main {json.dumps(main[0].get(key)) if main else '?'}")
    commits = Counter((c.get("git_commit", "?")[:7], c.get("git_dirty")) for c in crop)
    report.add("budget", "code version (commit, dirty)", None, str(dict(commits)))


# 5. Numbers ---------------------------------------------------------------------------------------

def check_numbers(report: Report, root: Path, n_boot: int) -> None:
    rows = []
    for sid, family in crop_scenarios():
        folder = root / CROP_PATH / family

        def run(folder=folder, sid=sid, family=family) -> None:
            for split in SPLITS:
                p = load_predictions(folder / f"{split}_predictions.csv", split)
                fresh = compute_metrics(p)
                saved = read_json(folder / f"{split}_metrics.json")
                off = [k for k, v in fresh.items() if not close(v, saved[k])]
                if split == "test":
                    if n_boot:
                        boot = saved["bootstrap"]
                        ci = case_bootstrap(p, boot["resamples"], boot["seed"])
                        off += [f"{m} CI" for m, v in ci.items() if not (
                            close(v["low"], saved["confidence_intervals"][m]["low"])
                            and close(v["high"], saved["confidence_intervals"][m]["high"]))]
                    rows.append(summary_row(family, saved))
                report.add("numbers", f"{sid} {split} metrics", not off,
                           f"macro-F1 {fresh['macro_f1']:.4f}" + (f"; differ: {off}" if off else ""))
        report.guard("numbers", f"{sid} crop/{family}", run)

    def tables() -> None:
        index = pd.read_csv(root / "evaluation_index.csv")
        report.add("numbers", "evaluation_index.csv", len(index) == 8 and (index["status"] == "evaluated").all()
                   and index["scenario_id"].tolist() == [sid for sid, _ in crop_scenarios()], f"{len(index)} rows")
        saved = pd.read_csv(root / "summary" / "crop_summary_table.csv")
        fresh = pd.DataFrame(rows)
        numeric = [c for c in fresh.columns if c not in ("scenario_id", "family")]
        off = [c for c in numeric if not np.allclose(saved[c], fresh[c], rtol=0, atol=TOLERANCE)]
        report.add("numbers", "crop_summary_table.csv",
                   saved["family"].tolist() == fresh["family"].tolist() and not off,
                   f"differ: {off}" if off else f"{len(saved)} rows, all columns")
    report.guard("numbers", "tables", tables)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = args.crop_root
    assignments = dict(pd.read_csv(args.split_assignments, dtype=str).itertuples(index=False, name=None))
    report = Report()

    check_files(report, root)
    check_test_set(report, root, assignments, args.crop_manifest)
    check_selection(report, root)
    report.guard("budget", "configs", lambda: check_budget(report, root, args.main_root))
    check_numbers(report, root, args.bootstrap)

    out = root / "verification"
    out.mkdir(exist_ok=True)
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
