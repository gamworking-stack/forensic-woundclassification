# Ticket T12: All 16 Scenarios

- **Ticket ID**: `T12`
- **Phase**: Phase 5 (Experiments)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §5 T12; [specs/phase3_training_code.md](../specs/phase3_training_code.md) §5.1
- **Status**: Completed (record written after the runs)

---

## 1. Goal

Run S01–S16 (8 families x full/ROI) under one split, seed and budget policy.

## 2. Runs

| Scenarios | Jobs | Dates | Notes |
|---|---|---|---|
| S13–S14 CNN-SVM | jobs 221593–221605 (`train_single`, `FAMILY=cnn_svm`); final results from 221604/221605 | 2026-10-01 | grid widened once on validation, then frozen (T06); reruns reuse cached features |
| S15–S16 AlexNet ensemble | jobs 221600–221613 (3 members + combine per path; full submitted twice, the second resumed); final full 221606–221609, ROI 221610–221613 | 2026-10-01 | seeds 43–45, representative seed 43 (T07) |
| S01–S12 single families | 36 grid runs + 12 select jobs | 2026-10-13 23:06 – 2026-10-14 00:40 | lr chosen on validation (T08b §7) |

- All 48 S01–S12 jobs completed on the first attempt (no timeout, failure or resubmission); 3.1 GPU-hours in total, 0.13–0.59 h of grid training per scenario.
- Code (`run_config.json` `git_commit`): S01–S12 `78a05ae`, S13–S14 `d7e041a`, S15–S16 `dccf3a4`. Every `run_config.json` says `git_dirty: true` because of an untracked `results/` folder on Raptor, not code changes (T09c §3).
- Same policy for every scenario: frozen case-level split, seed 42 (ensemble members 43–45), max 200 epochs with patience 25 on validation macro-F1, test predicted once with the selected checkpoint.

## 3. Optional 5-fold cross-validation

Not run. The ROADMAP lists it as optional ("if the data size allows"), on train+validation only, to report a mean and SD for the strongest models. Note that compute is not the obstacle (S01–S12 took 3.1 GPU-hours), and choosing the "strongest models" must not use the test results.

## 4. Acceptance Check (2026-10-14)

`evaluate.py` evaluated 16 of 16 scenarios, 0 missing, all on the same 157 test images from 62 cases (T08 §7). ✔
