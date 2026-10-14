# Ticket T00: Repository Skeleton and Requirements

- **Ticket ID**: `T00`
- **Phase**: Phase 0 (Environment)
- **Spec Reference**: [specs/phase0_environment.md](../specs/phase0_environment.md); [ROADMAP.md](../ROADMAP.md) §4, §5 T00
- **Status**: Completed (closed 2026-10-14; the skeleton was built with T01, this ticket records it and runs its check)

---

## 1. Goal

One repository layout and one dependency list that let the code run on the laptop and on Raptor.

## 2. What Was Built

The layout in ROADMAP §4, with two changes recorded in the spec §2: the model builder lives in `src/training/models.py` and the figures in `src/evaluation/plots.py`, so `src/models/` and `src/visualization/` stay empty; run outputs live on Raptor, not in `results/`.

## 3. Requirements

`requirements.txt` was trimmed on 2026-10-14 to the packages the code imports (`torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pillow`, `pytest`). Removed: `grad-cam` (T14 uses its own hook-based Grad-CAM), `seaborn`, `openpyxl`, `timm`, `tqdm` (never imported). Raptor jobs use the NGC container instead (T09b).

## 4. Acceptance Check (2026-10-14)

A new virtual environment on the laptop (`/usr/bin/python3` 3.9.6, `pip install -r requirements.txt` only), which resolved torch 2.8.0, torchvision 0.23.0, scikit-learn 1.6.1, pandas 2.3.3, numpy 2.0.2, matplotlib 3.9.4, pillow 11.3.0, pytest 8.4.2:

1. All 24 modules under `src/data`, `src/training` and `src/evaluation` import: ✔
2. `pytest tests/`: 106 passed: ✔
3. T00 ticked in `ROADMAP.md` §9: ✔
