# Ticket T09c: Data Transfer

- **Ticket ID**: `T09c`
- **Phase**: Phase 4 (HPC)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §5 T09c
- **Status**: Completed

---

## 1. Goal

Move code and annotations to Raptor and aggregate results back with `rsync`, keeping patient data on the cluster and out of git.

## 2. Requirements & Business Logic

1. **To Raptor** (`scripts/hpc/sync_to_raptor.sh`, laptop): upload the repo to `~/forensic-woundclassification` and the 7 CVAT exports to `/project/a260012_FORWCL/annotations/cvat_yolo_2026_09_30/`.
   - **New**: refuses to run with uncommitted changes (override `ALLOW_DIRTY=1` for a test), so every run's `run_config.json` names the exact commit.
2. **Raw images**: the 1,112 photographs were uploaded once to `/project/a260012_FORWCL/raw_data/wound_ml_7classes/` (5.7 GB) and are only read from there.
3. **Back to the laptop** (`scripts/hpc/fetch_results.sh [run]`, **new**): copy only aggregate outputs of a run (metrics JSON, class reports, confusion matrices, ROC curves, tuning and ensemble tables, training histories, run configs, `evaluation_index.csv`, `summary/`) into `../results_figures/<run>/`, outside the repo. Per-image prediction files, checkpoints, features and images are never copied.
4. **Git**: images, crops, checkpoints, zips and the local `results/` folder are ignored (`.gitignore`).

## 3. Finding: every run so far is flagged `git_dirty`

All `run_config.json` files record `git_dirty: true`. Cause, checked on Raptor 2026-10-14: the untracked `results/hpc_t03/` (2 files from the T03 run) in the Raptor checkout, which `git status --porcelain` counts. The S01–S12 code was synced from a clean `78a05ae`, so those runs ran exactly that commit. Fix: `results/` is now in `.gitignore`, and the sync refuses uncommitted changes. Report the recorded commit per scenario, not the dirty flag.

## 4. Acceptance Check (2026-10-14)

1. `sync_to_raptor.sh` uploads the repo and exports; with uncommitted changes it stops and lists them. ✔
2. `fetch_results.sh` copied 324 files (169 CSV, 90 JSON, 65 PNG) and 0 prediction, checkpoint or feature files. ✔
