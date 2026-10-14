# Ticket T16: Final Verification

- **Ticket ID**: `T16`
- **Phase**: Phase 6 (Figures and Results)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §5 T16, §7 (definition of done)
- **Scenarios**: all (S01–S16), plus the T12b CV run and the T14 Grad-CAM outputs
- **Status**: Completed

---

## 1. Goal

Before the results go into the thesis, confirm from the run files themselves (not from the tickets or logs) that 16 of 16 scenarios are present, all figures exist, the test set was never used for selection, and every reported number equals what the prediction files give.

---

## 2. Approach

One script, `src/evaluation/verify.py`, re-derives each answer independently of the code that produced it where possible, and reuses the original rule only where the rule itself is the thing being checked (best-epoch rule `EarlyStopping`, lr rule `select_lr.choose`, bootstrap functions with the recorded seed). It writes no file other than its own report and makes no change to the run.

| Check | What is re-derived |
|---|---|
| files | Per scenario: validation and test predictions, metrics JSON, class report, confusion matrix (CSV, PNG), ROC curves (CSV, PNG), `run_config.json`; the PNGs have a PNG signature. `evaluation_index.csv` and the 3 summary files |
| test set | Every case in a test prediction file is a test case in `data/splits/case_split_assignments.csv`, every validation case a validation case; all 16 scenarios share the same test images per case and class |
| selection | S01–S12: each grid run's best epoch replayed from its training history (and the validation prediction file equals that epoch's macro-F1); the tuning table equals the histories; the lr rule picks the row marked selected; unchosen runs have no test file and `skip_test: true`; the chosen run's test file is newer than `tuning_summary.csv`; the scenario's files are byte-identical copies of the chosen run's. S13–S14: the SVM table holds validation scores only, the rule picks the configured setting, test written after the table. S15–S16: seeds 43–45, one lr, member best epochs replayed, the Grad-CAM member chosen on validation |
| budget | All 42 trained networks (36 grid runs, 6 members) share `max_epochs`, `patience`, `batch_size`, `weight_decay`, `label_smoothing`, `full_scope`, `class_weights`, `pretrained`; grid runs use seed 42 |
| numbers | Validation and test metrics recomputed from the prediction files and compared with the JSON (tolerance 1e-9), test bootstrap intervals recomputed (2,000, seed 42); `evaluation_index.csv`, `summary_table.csv`, `full_vs_roi.csv` (all columns incl. intervals, p, Holm) recomputed; ensemble predictions equal the mean of the members |
| gradcam | The 14 examples are test images from 14 distinct cases, 2 per class; 7 × 2 × 14 maps; each map used the selected lr run or the representative member; 14 figures |
| cv | `cv_folds.csv` holds no test case; no CV folder has a test file; every CV training run ran in CV mode; each fold's cases lie in that fold; `cv_table.csv` and `cv_full_vs_roi.csv` (incl. intervals) recomputed; all 16 scenarios cover the same development cases |

Status per row: PASS, FAIL, or NOTE (recorded fact, not a pass/fail criterion). Exit code 1 on any FAIL.

---

## 3. Automated Testing (`tests/test_verify.py`, CPU)

A hand-built finished run (16 scenarios with grid runs, CNN-SVM, ensemble members, T08 and T15 outputs, Grad-CAM tables and a 2-fold CV run) passes with no FAIL. Each fault is caught, and only that fault:
- an unchosen lr run with a test file;
- a tuning table marking the wrong lr (also caught: table newer than the test file);
- one edited number in `full_vs_roi.csv`;
- an ensemble that is not the mean of its members;
- a missing scenario figure and a missing Grad-CAM figure;
- a test case listed as train in the split;
- a CV folder with a test file.

---

## 4. Files

| File | Purpose |
|---|---|
| `src/evaluation/verify.py` | CLI |
| `tests/test_verify.py` | Tests above |
| `scripts/hpc/fetch_results.sh` | Also copies `verification/` |
| `runbook.md` | Command and outputs |

---

## 5. Acceptance Check

1. `pytest tests/` passes.
2. On Raptor, `verify.py` on the main and CV runs reports no FAIL; every NOTE is explained here.
3. The numbers written in the tickets equal the result files.
4. Tick T16 in `ROADMAP.md` §9.

---

## 6. Run (2026-10-14)

`pytest tests/` passed locally (106 passed) and `tests/test_verify.py` in the Raptor container (8 passed). `verify.py --experiments-root .../experiments/main --cv-root .../experiments/cv` at commit `ae419ce` on a compute node, 9 min, exit 0:

| Check | PASS | FAIL | NOTE |
|---|---|---|---|
| files | 20 | 0 | 0 |
| test set | 33 | 0 | 0 |
| selection | 202 | 0 | 0 |
| budget | 8 | 0 | 1 |
| numbers | 40 | 0 | 0 |
| gradcam | 4 | 0 | 1 |
| cv | 6 | 0 | 0 |
| **Total** | **313** | **0** | **2** |

**NOTEs**
- **Code version**: the 42 main-run networks record commit `78a05ae` (36 grid runs) and `dccf3a4` (6 AlexNet members), all with `git_dirty: true`. The flag came from the untracked results folder `results/hpc_t03/` in the Raptor working copy, not from changed code (`git status --porcelain` counts untracked files). The folder was then added to `.gitignore` and `sync_to_raptor.sh` refuses uncommitted changes; the CV runs record `760e799` with `git_dirty: false` (T12b §7). Report the main-run commits in the thesis with this explanation.
- **Grad-CAM**: 195 of 196 recomputed predictions equal the prediction files; the exception is MobileNetV3-Small, full image, a cut wound (fp32 vs bf16 borderline case, T14 §6).

**Ticket numbers**: a one-off script compared the numbers written in T08 §7 (test macro-F1 and intervals), T15 §7 and T12b §7 (per-scenario CV table, paired table with p and Holm p, pooled secondary metrics) with the fetched result CSVs at 3 decimals: 88 numbers, 0 differ. The T14 §6 table (map share in the box, ratio to box area, images above uniform, correct per path) was recomputed from `gradcam_results.csv` on Raptor and matches.

**Not covered**: the verification checks files, so it cannot show that nobody looked at test results while choosing settings; the decision log is the tickets, where each analysis plan is dated before its results (T08b §2, T15 §2, T12b §2 states it is post hoc). Training was not rerun, so run-to-run training variation is not checked here (T12b §7 limits).
