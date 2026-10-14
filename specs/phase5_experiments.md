# Phase 5 Specification: Experiments

- **Phase**: 5 — Experiments
- **Related Tickets**: T11 (HPC smoke test), T12 (All 16 scenarios), T12b (5-fold cross-validation, optional)
- **Status**: Completed (as-built spec, written 2026-10-14 after the work; consolidates the tickets, which hold the details and results)
- **Owner**: Forensic Wound Classification Team (Preenapun Saokhieo)

---

## 1. Objectives

Run all 16 scenarios (8 families x 2 input paths) under one split, seed and budget policy, after a smoke test on real images, and optionally repeat them by cross-validation on the development data.

---

## 2. Shared Policy (every scenario)

| Item | Value |
|---|---|
| Split | frozen case-level split (Phase 1), matched 1,100-image scope for both paths (Phase 2) |
| Budget | max 200 epochs, patience 25 on validation macro-F1, batch 32, AdamW (weight decay 1e-4), label smoothing 0.05, class weights |
| Seeds | 42; AlexNet ensemble members 43, 44, 45 |
| Selection | on validation only: best epoch per run; learning rate from {1e-5, 3e-5, 1e-4} for S01–S12 (T08b); SVM setting for S13–S14 (T06) |
| Test | predicted once per scenario, after selection, by the selected weights |

---

## 3. Components

### 3.1 T11 — Smoke test

EfficientNet-B0, 2 epochs on the real images, full and ROI, through `train_single.sbatch` without code changes (`scripts/hpc/submit_t11_smoke.sh`). Passed 2026-10-01: jobs `221591` and `221592`, 779 / 164 / 157 images on both paths (`runbook.md`, T05 training library).

### 3.2 T12 — All 16 scenarios

`submit_all.sh` on Raptor. S13–S16 ran 2026-10-01; S01–S12 ran 2026-10-13 to 14 as 36 grid runs and 12 select jobs, all 48 completed on the first attempt (3.1 GPU-hours). Code commits per scenario are recorded in `run_config.json` (T12 §2).

### 3.3 T12b — Cross-validation (optional, supplementary)

5 case-grouped folds over the 356 development cases (`data/splits/cv_folds.csv`); every scenario reuses its main-run settings with a fixed epoch count and never loads the test split; 110 jobs. Decided after the test results were seen, so it is reported as post hoc and changes no model or test result.

---

## 4. Acceptance Checks

1. T11: both smoke runs finish and write all output files.
2. T12: 16 of 16 scenarios have validation and test predictions; T08b leaves exactly one test file per S01–S12 scenario, in the selected run.
3. T12b: every planned job finishes; each scenario's 5 fold files cover every development case once.

All met (T12 §4, T12b §7). Checked again from the run files in T16.
