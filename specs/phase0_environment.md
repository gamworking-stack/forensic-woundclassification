# Phase 0 Specification: Environment and Repository Skeleton

- **Phase**: 0 — Environment
- **Related Tickets**: T00 (Skeleton and requirements)
- **Status**: Completed (as-built spec, written 2026-10-14 after the work; consolidates `tickets/T00_skeleton.md`)
- **Owner**: Forensic Wound Classification Team (Preenapun Saokhieo)

---

## 1. Objectives

Give every later phase one repository layout and one list of Python dependencies, so that the code runs the same on the laptop (tests, dry runs) and on Raptor (real runs).

---

## 2. Repository Layout (as built)

| ROADMAP §4 | As built | Note |
|---|---|---|
| `src/data` | `src/data/` | metadata validation, case split, ROI crops, audit, CV folds |
| `src/models` | `src/training/models.py` | the model builder sits with the training code that uses it; `src/models/` is unused |
| `src/training` | `src/training/` | shared library, CNN-SVM, ensemble, lr selection, CV plan |
| `src/evaluation` | `src/evaluation/` | metrics, evaluation, summary, CV summary, Grad-CAM, verification |
| `src/visualization` | `src/evaluation/plots.py` | figures are drawn by the evaluation code; `src/visualization/` is unused |
| `scripts/slurm/` | `scripts/slurm/` (`.sbatch`) and `scripts/hpc/` (submit, sync, fetch, watch) | |
| `data/` | `data/` | manifests and reports only; images stay on Raptor |
| `results/` | `/project/a260012_FORWCL/experiments/<run>/` on Raptor; aggregates in `../results_figures/<run>/` | patient-derived outputs are not kept in the repo (`results/` is git-ignored) |
| `specs/ tickets/ logs/` | `specs/`, `tickets/`, `logs/` (SLURM logs go to `logs/` on Raptor) | |
| `runbook.md ROADMAP.md requirements.txt` | present | plus `docs/diagrams/` (Archify) and `tests/` |

---

## 3. Dependencies

`requirements.txt` lists exactly the third-party packages the code imports: `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pillow`, and `pytest` for the tests, as minimum versions.

- **Dropped from the ROADMAP list**: `grad-cam` (T14 implements Grad-CAM with forward/backward hooks, which also handles the ViT-B/16 token reshape), and the unused `seaborn`, `openpyxl`, `timm` and `tqdm`.
- **On Raptor** no environment is installed: every job runs in the NGC container `/common/sif/pytorch-24.11-py3.sif` (Python 3.12, torch 2.6, torchvision 0.20, CUDA 12.6), which provides all of these (Phase 4, T09b). Each run records the versions it used in `run_config.json`.

---

## 4. Acceptance Check

1. In a fresh virtual environment created from `requirements.txt` alone, every module under `src/data`, `src/training` and `src/evaluation` imports.
2. `pytest tests/` passes in that environment.
3. Tick T00 in `ROADMAP.md` §9.

Result: see `tickets/T00_skeleton.md` §4.
