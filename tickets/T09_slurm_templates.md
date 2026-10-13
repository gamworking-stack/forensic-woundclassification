# Ticket T09: SLURM Templates and Submission

- **Ticket ID**: `T09`
- **Phase**: Phase 4 (HPC)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §5 T09
- **Status**: Completed (written after the fact; the templates were built with T03, T05–T08b)

---

## 1. Goal

One SLURM template per kind of job, and one command that submits every unfinished scenario, skips finished or queued ones and resumes interrupted ones.

## 2. What Was Built

| File | Job | Partition | Resources | Used by |
|---|---|---|---|---|
| `scripts/slurm/train_single.sbatch` | one family x one input path (`FAMILY=cnn_svm` runs `cnn_svm.py`), `--resume` | `gpu` | 1 GPU, 12 CPUs (`--num-workers` = CPUs - 2), 96 GB, default 2 days | T05, T06, T07, T08b, T11 |
| `scripts/slurm/ensemble_combine.sbatch` | soft vote of 3 members | `compute` | 2 CPUs, 8 GB, 30 min | T07 |
| `scripts/slurm/tune_select.sbatch` | pick the learning rate, predict test | `gpu` | 1 GPU, 12 CPUs, 96 GB, 2 h | T08b |
| `scripts/slurm/t03_roi_crops.sbatch` | ROI crops + comparison | `compute` | 2 CPUs, 16 GB, 2 h | T03 |
| `scripts/hpc/submit_all.sh` | S01–S16: skip finished (`test_predictions.csv`) or queued (job name); S01–S12 as 3 grid runs + `afterok` select job; S15/S16 through `submit_alexnet_ensemble.sh` | – | `--time` 12 h per training run, 4 h CNN-SVM | T12 |
| `scripts/hpc/submit_alexnet_ensemble.sh` | 3 members + `afterok` combine, `--kill-on-invalid-dep=yes` | – | 6 h per member | T07 |
| `scripts/hpc/submit_t11_smoke.sh` | 2-epoch smoke run, full and ROI | – | 1 h | T11 |
| `scripts/hpc/watch_jobs.sh` | follow `.out`/`.err` live, `sacct` summary at the end | – | – | all |

**Deviation from the ROADMAP**: CNN-SVM runs on `gpu` (feature extraction needs the GPU; the SVM grid takes minutes), not on `compute`. Every job runs inside the Apptainer container (T09b).

## 3. Acceptance Check (2026-10-14)

1. `sbatch --test-only` passes for every template on Raptor: `train_single` (EfficientNet-B0 ROI and CNN-SVM full) and `tune_select` on `gpu`, `ensemble_combine` and `t03_roi_crops` on `compute`. ✔
2. `submit_all.sh` skip, queue and dependency logic checked against stub `sbatch`/`squeue` (58 jobs on an empty run; finished and queued grid runs skipped or waited on). ✔
3. Real use: all 48 S01–S12 jobs, the S13–S16 jobs and the T03/T11 jobs completed through these templates (T12). ✔
