# Phase 4 Specification: HPC (MedCMU Raptor)

- **Phase**: 4 — HPC
- **Related Tickets**: T09 (SLURM templates), T09b (Cluster environment), T09c (Data transfer), T10 (Runbook)
- **Status**: Completed (as-built spec, written 2026-10-14 after the work; consolidates the four tickets, which hold the details and acceptance results)
- **Owner**: Forensic Wound Classification Team (Preenapun Saokhieo)

---

## 1. Objectives

Run every scenario on Raptor through a small set of SLURM templates, in one fixed software environment, with data and results moving only where the privacy rules allow, and with a runbook that lets another person repeat every step.

---

## 2. Cluster Facts (resolved from ROADMAP §2b)

| Item | Value |
|---|---|
| Login | `ssh preenapun.s@raptor.med.cmu.ac.th`, reachable from the CMU network or VPN only; key login from the laptop |
| Account / project | `a260012`, project root `/project/a260012_FORWCL` |
| Partitions | `gpu` (2 nodes x 2 H100 PCIe), `compute` (CPU), `short` (tiny tests) |
| GPU request | `--gres=gpu:1`, one H100 per job (training is single-GPU) |
| Software | `module load apptainer/1.3.5`; container `/common/sif/pytorch-24.11-py3.sif` |
| Pretrained weights | downloaded once on the login node into `TORCH_HOME=/project/a260012_FORWCL/torch_cache` |

---

## 3. Components

### 3.1 T09 — SLURM templates (`scripts/slurm/`, `scripts/hpc/`)

- `train_single.sbatch`: one family x one input path (`FAMILY=cnn_svm` runs `cnn_svm.py`), always `--resume`, `gpu`, 1 GPU, 12 CPUs, 96 GB.
- `tune_select.sbatch`: T08b learning-rate choice and the one test pass, `gpu`.
- `ensemble_combine.sbatch`: soft vote, `compute`. `t03_roi_crops.sbatch`: ROI crops, `compute`.
- `submit_all.sh`: S01–S16; skips finished (`test_predictions.csv`) or queued (job name) scenarios; S01–S12 as 3 grid runs plus an `afterok` select job; S15–S16 through `submit_alexnet_ensemble.sh`. `submit_cv.sh` does the same for the T12b plan.
- Deviation from the ROADMAP: CNN-SVM runs on `gpu`, since its feature extraction needs the GPU.

### 3.2 T09b — Environment

The NGC container replaces the planned micromamba environment: nothing to install, the same image for every job, versions recorded per run.

### 3.3 T09c — Data transfer

- To Raptor: `sync_to_raptor.sh` uploads the committed repo (refuses uncommitted changes) and the 7 CVAT exports. Raw images were uploaded once to `raw_data/wound_ml_7classes/`.
- Back to the laptop: `fetch_results.sh [run]` copies aggregate outputs only (metrics, reports, figures, tables, configs, `summary/`, `verification/`) to `../results_figures/<run>/`, outside the repo.
- Never copied off Raptor or committed: images, crops, checkpoints, features, per-image prediction files, Grad-CAM overlays.

### 3.4 T10 — Runbook (`runbook.md`)

Connection, environment, data locations, submit, monitor, resume, results, support contact, plus one section per analysis step (T08, T15, T14, T12b, T16) and the diagrams.

---

## 4. Acceptance Checks

1. `sbatch --test-only` passes for every template (T09 §3).
2. `torch.cuda.is_available()` on the GPU nodes; container versions recorded in every `run_config.json` (T09b §3).
3. Sync refuses a dirty tree; fetch copies no per-image file (T09c §4).
4. Every runbook command was used for the reported results (T10 §3).

All four met; see each ticket. Open finding carried into the thesis: the main-run configs are flagged `git_dirty` because of an untracked results folder on Raptor, not changed code (T09c §3, T16 §6).
