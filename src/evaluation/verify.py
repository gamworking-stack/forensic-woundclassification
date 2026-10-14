#!/usr/bin/env python3
"""Final verification of a finished run (T16): every check re-derives its answer from the run files.

1. files:      16 of 16 scenarios hold their prediction, metric, report and figure files.
2. test set:   test predictions contain only test cases of the frozen split, validation predictions
               only validation cases, and all 16 scenarios share one test set.
3. selection:  every choice is reproduced from validation data alone (best epoch from the training
               history, learning rate from the tuning table, SVM setting, ensemble members), only the
               selected run has test predictions, and those were written after the choice.
4. budget:     every trained network shares one training budget and seed policy.
5. numbers:    test/validation metrics, evaluation_index.csv, summary_table.csv and full_vs_roi.csv
               are recomputed from the prediction files (bootstrap intervals with the same seed).
6. gradcam:    the fixed examples are test images, every map exists, maps used the reported models.
7. cv:         (with --cv-root) no CV run loads the test split, folds match cv_folds.csv, and the CV
               table and paired comparison are recomputed.

Writes <experiments-root>/verification/verification_report.csv (no image paths) and exits 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.evaluate import FAMILY_ORDER, INPUT_PATHS, SCENARIO_IDS, SPLITS  # noqa: E402
from src.evaluation.gradcam import FAMILIES as GRADCAM_FAMILIES  # noqa: E402
from src.evaluation.metrics import case_bootstrap, compute_metrics, load_predictions, summary_metrics  # noqa: E402
from src.evaluation.summarize import load_pairs, paired_comparison, summary_table  # noqa: E402
from src.training.engine import EarlyStopping  # noqa: E402
from src.training.select_lr import LR_GRID, choose  # noqa: E402


SINGLE_FAMILIES = tuple(f for f in FAMILY_ORDER if f not in ("cnn_svm", "alexnet_ensemble"))
SCENARIO_FILES = tuple(f"{split}_{name}" for split in SPLITS for name in (
    "predictions.csv", "metrics.json", "classification_report.csv",
    "confusion_matrix.csv", "confusion_matrix.png", "roc_curves.csv", "roc_curves.png",
)) + ("run_config.json",)
BUDGET_KEYS = ("max_epochs", "patience", "batch_size", "weight_decay", "label_smoothing",
               "full_scope", "class_weights", "pretrained")
ENSEMBLE_SEEDS = (43, 44, 45)
TOLERANCE = 1e-9


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final verification of a finished run (T16).")
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, default=Path("data/splits/case_split_assignments.csv"))
    parser.add_argument("--cv-root", type=Path, default=None, help="T12b run; omitted: CV not checked.")
    parser.add_argument("--cv-folds", type=Path, default=Path("data/splits/cv_folds.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Paired resamples used by summarize.py/cv_summary.py; 0 skips the interval check.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


class Report:
    """Rows of (check, item, status, detail); status is PASS, FAIL or NOTE."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []

    def add(self, check: str, item: str, ok: Optional[bool], detail: str = "") -> None:
        status = "NOTE" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append({"check": check, "item": item, "status": status, "detail": detail})

    def guard(self, check: str, item: str, fn) -> None:
        """Run one check; an exception (missing file, bad format) is a FAIL, not a crash."""
        try:
            fn()
        except Exception as error:  # noqa: BLE001 - every failure belongs in the report
            self.add(check, item, False, f"{type(error).__name__}: {error}")

    @property
    def failed(self) -> List[Dict[str, str]]:
        return [r for r in self.rows if r["status"] == "FAIL"]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: Any, b: Any) -> bool:
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= TOLERANCE


def scenarios() -> Iterable[tuple]:
    for (input_path, family), scenario_id in sorted(SCENARIO_IDS.items(), key=lambda kv: kv[1]):
        yield scenario_id, input_path, family


def best_epoch(history: pd.DataFrame) -> tuple:
    """Replay the training rule (engine.EarlyStopping) on a history: (best epoch, its macro-F1, its loss)."""
    stopper = EarlyStopping(patience=len(history) + 1)
    for row in history.to_dict("records"):
        stopper.step(int(row["epoch"]), float(row["validation_macro_f1"]), float(row["validation_loss"]))
    return stopper.best_epoch, stopper.best_f1, stopper.best_loss


def check_history(report: Report, item: str, folder: Path, config: Dict[str, Any],
                  expected_epoch: Optional[int] = None, expected_f1: Optional[float] = None) -> None:
    """Best epoch chosen on validation, early stopping consistent, validation file = best epoch."""
    history = pd.read_csv(folder / "training_history.csv")
    epoch, f1, _ = best_epoch(history)
    epochs = len(history)
    stopped_right = epochs == config["max_epochs"] or epochs - epoch == config["patience"]
    detail = f"best epoch {epoch} of {epochs}, validation macro-F1 {f1:.6f}"
    ok = stopped_right and (expected_epoch is None or expected_epoch == epoch) \
        and (expected_f1 is None or close(expected_f1, f1))
    if not ok:
        detail += f" (recorded epoch {expected_epoch}, F1 {expected_f1}; stop rule held: {stopped_right})"
    report.add("selection", f"{item}: best epoch", ok, detail)
    val = compute_metrics(load_predictions(folder / "validation_predictions.csv", "validation"))["macro_f1"]
    report.add("selection", f"{item}: validation file is the best epoch", close(val, f1),
               f"recomputed {val:.6f} vs history {f1:.6f}")


# 1. Files -----------------------------------------------------------------------------------------

def check_files(report: Report, root: Path) -> None:
    for scenario_id, input_path, family in scenarios():
        folder = root / input_path / family
        missing = [n for n in SCENARIO_FILES if not (folder / n).is_file()]
        bad_png = [n for n in SCENARIO_FILES if n.endswith(".png") and n not in missing
                   and (folder / n).read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"]
        report.add("files", f"{scenario_id} {input_path}/{family}", not missing and not bad_png,
                   f"missing {missing}, not PNG {bad_png}" if missing or bad_png
                   else f"{len(SCENARIO_FILES)} files incl. 4 figures")
    for name in ("evaluation_index.csv", "summary/summary_table.csv", "summary/full_vs_roi.csv",
                 "summary/full_vs_roi_macro_f1.png"):
        report.add("files", name, (root / name).is_file())


# 2. Test set --------------------------------------------------------------------------------------

def check_test_set(report: Report, root: Path, assignments: Dict[str, str]) -> None:
    signatures = {}
    for scenario_id, input_path, family in scenarios():
        folder = root / input_path / family

        def run(folder=folder, scenario_id=scenario_id, input_path=input_path, family=family) -> None:
            for split in SPLITS:
                p = load_predictions(folder / f"{split}_predictions.csv", split)
                wrong = sorted({c for c in p.cases if assignments.get(c) != split})
                report.add("test set", f"{scenario_id} {split} cases", not wrong,
                           f"{len(set(p.cases))} cases, {len(p.cases)} images"
                           + (f"; not {split} in the frozen split: {wrong[:5]}" if wrong else ""))
                if split == "test":
                    signatures[f"{input_path}/{family}"] = p.signature()
        report.guard("test set", f"{scenario_id} {input_path}/{family}", run)
    if signatures:
        reference = next(iter(signatures.values()))
        different = [k for k, v in signatures.items() if v != reference]
        report.add("test set", "same test images in every scenario", not different and len(signatures) == 16,
                   f"{len(signatures)} scenarios" + (f"; differ: {different}" if different else ""))


# 3. Selection -------------------------------------------------------------------------------------

def check_lr_selection(report: Report, folder: Path, item: str) -> Path:
    """T08b: tuning table reproduced from the grid runs, test only for the chosen run, after the choice."""
    tuning = pd.read_csv(folder / "tuning_summary.csv", dtype={"lr": str})
    report.add("selection", f"{item}: lr grid", tuple(tuning["lr"]) == LR_GRID, f"{list(tuning['lr'])}")
    runs = []
    for row in tuning.to_dict("records"):
        grid = folder / f"lr_{row['lr']}"
        config = read_json(grid / "run_config.json")
        report.add("selection", f"{item}: lr_{row['lr']} trained with lr {row['lr']}",
                   float(config["lr"]) == float(row["lr"]) and config["seed"] == 42, f"seed {config['seed']}")
        check_history(report, f"{item} lr_{row['lr']}", grid, config,
                      int(row["best_epoch"]), float(row["validation_macro_f1"]))
        runs.append({"lr": row["lr"], "validation_macro_f1": row["validation_macro_f1"],
                     "validation_loss": row["validation_loss"], "dir": grid, "config": config})
    winner = choose(runs)
    flagged = tuning.index[tuning["selected"].astype(str) == "True"].tolist()
    report.add("selection", f"{item}: chosen lr reproduced", flagged == [winner],
               f"rule picks {runs[winner]['lr']}, table marks {[runs[i]['lr'] for i in flagged]}")

    chosen = runs[winner]["dir"]
    for i, run in enumerate(runs):
        has_test = (run["dir"] / "test_predictions.csv").exists()
        if i == winner:
            ok = has_test and (run["dir"] / "test_predictions.csv").stat().st_mtime \
                >= (folder / "tuning_summary.csv").stat().st_mtime
            report.add("selection", f"{item}: test predicted after the lr choice", ok)
        else:
            report.add("selection", f"{item}: lr_{run['lr']} never predicted test",
                       not has_test and run["config"].get("skip_test") is True,
                       f"test file present: {has_test}, skip_test: {run['config'].get('skip_test')}")
    same = [n for n in ("run_config.json", "training_history.csv", "validation_predictions.csv",
                        "test_predictions.csv") if (folder / n).read_bytes() == (chosen / n).read_bytes()]
    report.add("selection", f"{item}: scenario files are the chosen run's", len(same) == 4, f"identical: {same}")
    return chosen


def check_svm_selection(report: Report, folder: Path, item: str) -> None:
    tuning = pd.read_csv(folder / "svm_validation_tuning.csv", keep_default_na=False)
    config = read_json(folder / "run_config.json")
    extra = [c for c in tuning.columns if c not in ("kernel", "C", "gamma") and not c.startswith("validation_")]
    report.add("selection", f"{item}: tuning table has validation scores only", not extra, f"other columns {extra}")
    best = max(range(len(tuning)), key=lambda i: (tuning["validation_macro_f1"][i], -i))
    row, params = tuning.iloc[best], config["best_params"]
    same = row["kernel"] == params["kernel"] and float(row["C"]) == float(params["C"]) \
        and str(row["gamma"]) == str(params.get("gamma", ""))
    report.add("selection", f"{item}: chosen SVM reproduced",
               same and close(row["validation_macro_f1"], config["best_validation_macro_f1"]),
               f"{len(tuning)} settings; rule picks {row['kernel']} C={row['C']} gamma={row['gamma']}, config {params}")
    report.add("selection", f"{item}: test predicted after the SVM choice",
               (folder / "test_predictions.csv").stat().st_mtime >= (folder / "svm_validation_tuning.csv").stat().st_mtime)


def check_ensemble(report: Report, folder: Path, item: str) -> Path:
    members = pd.read_csv(folder / "ensemble_members.csv")
    configs = []
    for row in members.to_dict("records"):
        member = folder / f"member_{int(row['member']):02d}"
        config = read_json(member / "run_config.json")
        configs.append(config)
        check_history(report, f"{item} member_{int(row['member']):02d}", member, config,
                      int(row["best_epoch"]), float(row["best_validation_macro_f1"]))
    seeds = tuple(c["seed"] for c in configs)
    report.add("selection", f"{item}: fixed members, no selection", seeds == ENSEMBLE_SEEDS
               and len({float(c["lr"]) for c in configs}) == 1,
               f"seeds {seeds}, lr {sorted({c['lr'] for c in configs})}")
    rep = max(range(len(members)), key=lambda i: (members["best_validation_macro_f1"][i], -members["seed"][i]))
    report.add("selection", f"{item}: Grad-CAM member chosen on validation",
               members["representative"].astype(str).tolist() == [str(i == rep) for i in range(len(members))],
               f"member {rep + 1}")
    for split in SPLITS:
        voted = pd.read_csv(folder / f"{split}_predictions.csv")
        frames = [pd.read_csv(folder / f"member_{int(m):02d}" / f"{split}_predictions.csv") for m in members["member"]]
        prob = [c for c in voted.columns if c.startswith("prob_")]
        mean = np.mean([f[prob].to_numpy() for f in frames], axis=0)
        same_rows = all(f["relative_path"].equals(voted["relative_path"]) for f in frames)
        report.add("numbers", f"{item}: {split} = mean of members",
                   same_rows and np.allclose(mean, voted[prob].to_numpy(), rtol=0, atol=1e-12))
    return folder / f"member_{rep + 1:02d}"


def check_selection(report: Report, root: Path) -> Dict[str, Path]:
    """Returns the model folder behind each scenario (for the Grad-CAM check)."""
    chosen: Dict[str, Path] = {}
    for scenario_id, input_path, family in scenarios():
        folder, item = root / input_path / family, f"{scenario_id} {input_path}/{family}"

        def run(folder=folder, item=item, key=f"{input_path}/{family}", family=family) -> None:
            if family in SINGLE_FAMILIES:
                chosen[key] = check_lr_selection(report, folder, item)
            elif family == "cnn_svm":
                check_svm_selection(report, folder, item)
            else:
                chosen[key] = check_ensemble(report, folder, item)
        report.guard("selection", item, run)
    return chosen


# 4. Budget ----------------------------------------------------------------------------------------

def check_budget(report: Report, root: Path) -> None:
    configs = {}
    for _, input_path, family in scenarios():
        base = root / input_path / family
        pattern = "lr_*/run_config.json" if family in SINGLE_FAMILIES else "member_*/run_config.json"
        for path in sorted(base.glob(pattern)):
            configs[str(path.parent.relative_to(root))] = read_json(path)
    for key in BUDGET_KEYS:
        values = Counter(json.dumps(c.get(key)) for c in configs.values())
        report.add("budget", key, len(values) == 1, f"{dict(values)} over {len(configs)} trained networks")
    commits = Counter((c.get("git_commit", "?")[:7], c.get("git_dirty")) for c in configs.values())
    report.add("budget", "code version (commit, dirty)", None, str(dict(commits)))


# 5. Numbers ---------------------------------------------------------------------------------------

def check_numbers(report: Report, root: Path, n_boot: int, seed: int) -> None:
    recomputed: Dict[tuple, Dict[str, Any]] = {}
    for scenario_id, input_path, family in scenarios():
        folder = root / input_path / family

        def run(folder=folder, scenario_id=scenario_id, input_path=input_path, family=family) -> None:
            for split in SPLITS:
                p = load_predictions(folder / f"{split}_predictions.csv", split)
                fresh = compute_metrics(p)
                saved = read_json(folder / f"{split}_metrics.json")
                off = [k for k, v in fresh.items() if not close(v, saved[k])]
                if split == "test":
                    recomputed[(input_path, family)] = fresh
                    if n_boot:
                        boot = saved["bootstrap"]
                        ci = case_bootstrap(p, boot["resamples"], boot["seed"])
                        off += [f"{m} CI" for m, v in ci.items() if not (
                            close(v["low"], saved["confidence_intervals"][m]["low"])
                            and close(v["high"], saved["confidence_intervals"][m]["high"]))]
                report.add("numbers", f"{scenario_id} {split} metrics", not off,
                           f"macro-F1 {fresh['macro_f1']:.4f}" + (f"; differ: {off}" if off else ""))
        report.guard("numbers", f"{scenario_id} {input_path}/{family}", run)

    def tables() -> None:
        index = pd.read_csv(root / "evaluation_index.csv")
        off = [r["scenario_id"] for r in index.to_dict("records")
               if not close(r["test_macro_f1"], recomputed[(r["input_path"], r["family"])]["macro_f1"])]
        report.add("numbers", "evaluation_index.csv", len(index) == 16 and not off and
                   (index["status"] == "evaluated").all(), f"differ: {off}" if off else "16 evaluated")

        saved = pd.read_csv(root / "summary" / "summary_table.csv")
        fresh = summary_table(root)
        numeric = [c for c in fresh.columns if c not in ("scenario_id", "family", "input_path")]
        off = [c for c in numeric if not np.allclose(saved[c], fresh[c], rtol=0, atol=TOLERANCE)]
        f1_off = [r["scenario_id"] for r in saved.to_dict("records")
                  if not close(r["macro_f1"], recomputed[(r["input_path"], r["family"])]["macro_f1"])]
        report.add("numbers", "summary_table.csv", saved[["scenario_id", "family", "input_path"]].equals(
            fresh[["scenario_id", "family", "input_path"]]) and not off and not f1_off,
            f"differ: {off + f1_off}" if off or f1_off else "16 rows, all columns")

        saved = pd.read_csv(root / "summary" / "full_vs_roi.csv")
        compare_paired(report, "full_vs_roi.csv", saved, load_pairs(root), n_boot, seed)
    report.guard("numbers", "tables", tables)


def compare_paired(report: Report, item: str, saved: pd.DataFrame, pairs, n_boot: int, seed: int) -> None:
    """Point estimates always; intervals, p-values and Holm when n_boot > 0."""
    columns = ["full", "roi", "difference"]
    if n_boot:
        fresh = paired_comparison(pairs, n_boot, seed)
        columns += ["ci_low", "ci_high", "p_value", "p_holm", "resamples"]
    else:
        fresh = paired_comparison(pairs, 1, seed)
    same_keys = saved[["family", "metric"]].equals(fresh[["family", "metric"]])
    off = [c for c in columns if not np.allclose(saved[c], fresh[c], rtol=0, atol=TOLERANCE, equal_nan=True)]
    report.add("numbers", item, same_keys and not off,
               (f"differ: {off}" if off else f"{len(saved)} rows, columns {columns}")
               + ("" if n_boot else "; intervals not rechecked (--bootstrap 0)"))


# 6. Grad-CAM --------------------------------------------------------------------------------------

def check_gradcam(report: Report, root: Path, assignments: Dict[str, str], chosen: Dict[str, Path]) -> None:
    out = root / "gradcam"
    examples = pd.read_csv(out / "examples.csv", dtype=str, keep_default_na=False)
    per_class = examples.groupby("canonical_wound_label").size()
    ok = (examples["split"] == "test").all() and all(assignments.get(c) == "test" for c in examples["pseudonimized_id"]) \
        and not examples["pseudonimized_id"].duplicated().any() and per_class.nunique() == 1
    report.add("gradcam", "examples are test images from distinct cases", ok,
               f"{len(examples)} images, {per_class.min()} per class, {examples['pseudonimized_id'].nunique()} cases")

    results = pd.read_csv(out / "gradcam_results.csv", dtype={"case": str})
    expected = len(GRADCAM_FAMILIES) * len(INPUT_PATHS) * len(examples)
    report.add("gradcam", "one map per family, path and example", len(results) == expected and
               not results.duplicated(["family", "input_path", "image"]).any(), f"{len(results)} of {expected}")
    wrong_model = []
    for (family, input_path), rows in results.groupby(["family", "input_path"]):
        model = chosen.get(f"{input_path}/{family}")
        if model is None or not rows["checkpoint"].map(lambda c: Path(c).parent.parts[-3:] == model.parts[-3:]).all():
            wrong_model.append(f"{input_path}/{family}")
    report.add("gradcam", "maps use the reported models", not wrong_model, f"differ: {wrong_model}" if wrong_model
               else "selected lr run or representative member")
    mismatched = results.loc[~results["matches_prediction_file"].astype(bool), ["family", "input_path", "class"]]
    report.add("gradcam", "predictions equal the prediction files", None,
               f"{len(results) - len(mismatched)} of {len(results)}"
               + "".join(f"; differs: {f} {i} {c}" for f, i, c in mismatched.itertuples(index=False)))
    figures = [f"{family}_set{k}.png" for family in GRADCAM_FAMILIES for k in sorted(examples["set"].unique())]
    missing = [f for f in figures if not (out / f).is_file() or (out / f).read_bytes()[:4] != b"\x89PNG"]
    report.add("gradcam", "figures", not missing, f"missing {missing}" if missing else f"{len(figures)} figures")


# 7. Cross-validation ------------------------------------------------------------------------------

def check_cv(report: Report, cv_root: Path, cv_folds: Path, folds: int, assignments: Dict[str, str],
             n_boot: int, seed: int) -> None:
    fold_of = dict(pd.read_csv(cv_folds, dtype=str).itertuples(index=False, name=None))
    leaked = sorted(c for c in fold_of if assignments.get(c) == "test")
    report.add("cv", "cv_folds.csv has no test case", not leaked and all(assignments.get(c) for c in fold_of),
               f"{len(fold_of)} cases in {len(set(fold_of.values()))} folds")
    test_files = sorted(str(p.relative_to(cv_root)) for p in cv_root.rglob("test_predictions.csv"))
    report.add("cv", "no CV run predicted test", not test_files, f"found {test_files[:5]}" if test_files else "")
    not_cv = sorted(str(p.parent.relative_to(cv_root)) for p in cv_root.rglob("run_config.json")
                    if read_json(p).get("family") != "ensemble" and not read_json(p).get("cv_folds"))
    report.add("cv", "every CV run ran in CV mode (test rows dropped)", not not_cv, f"not CV: {not_cv[:5]}")

    def run() -> None:
        saved = pd.read_csv(cv_root / "summary" / "cv_table.csv").set_index("scenario_id")
        pairs, off = {}, []
        with tempfile.TemporaryDirectory() as tmp:
            for scenario_id, input_path, family in scenarios():
                frames, per_fold = [], []
                for k in range(1, folds + 1):
                    path = cv_root / input_path / family / f"fold_{k}" / "validation_predictions.csv"
                    p = load_predictions(path, "validation")
                    outside = sorted({c for c in p.cases if fold_of.get(c) != str(k)})
                    if outside:
                        off.append(f"{scenario_id} fold {k}: cases outside the fold {outside[:3]}")
                    per_fold.append(summary_metrics(p.y_true, p.y_pred, p.probs))
                    frames.append(pd.read_csv(path, dtype=str))
                joined = Path(tmp) / f"{scenario_id}.csv"
                pd.concat(frames, ignore_index=True).to_csv(joined, index=False)
                oof = load_predictions(joined, "validation")
                pooled = compute_metrics(oof)
                row = saved.loc[scenario_id]
                for m in ("macro_f1", "balanced_accuracy", "macro_auc", "accuracy"):
                    values = np.array([f[m] for f in per_fold])
                    if not (close(row[f"{m}_mean"], values.mean()) and close(row[f"{m}_sd"], values.std(ddof=1))
                            and close(row[f"{m}_oof"], pooled[m])):
                        off.append(f"{scenario_id} {m}")
                pairs.setdefault(family, {})[input_path] = oof
        report.add("cv", "cv_table.csv (fold mean, SD, out of fold)", not off, f"differ: {off[:5]}" if off
                   else f"16 scenarios x {folds} folds")
        cases = {frozenset(p.cases) for d in pairs.values() for p in d.values()}
        report.add("cv", "every scenario covers the same development cases", len(cases) == 1,
                   f"{len(next(iter(cases)))} cases" if len(cases) == 1 else f"{len(cases)} different case sets")
        ordered = {f: tuple(pairs[f][p] for p in INPUT_PATHS) for f in FAMILY_ORDER}
        compare_paired(report, "cv_full_vs_roi.csv", pd.read_csv(cv_root / "summary" / "cv_full_vs_roi.csv"),
                       ordered, n_boot, seed)
    report.guard("cv", "recompute", run)
    report.add("cv", "cv_full_vs_roi_macro_f1.png", (cv_root / "summary" / "cv_full_vs_roi_macro_f1.png").is_file())


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = args.experiments_root
    assignments = dict(pd.read_csv(args.split_assignments, dtype=str).itertuples(index=False, name=None))
    report = Report()

    check_files(report, root)
    check_test_set(report, root, assignments)
    chosen = check_selection(report, root)
    report.guard("budget", "configs", lambda: check_budget(report, root))
    check_numbers(report, root, args.bootstrap, args.seed)
    report.guard("gradcam", "outputs", lambda: check_gradcam(report, root, assignments, chosen))
    if args.cv_root is not None:
        check_cv(report, args.cv_root, args.cv_folds, args.folds, assignments, args.bootstrap, args.seed)

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
