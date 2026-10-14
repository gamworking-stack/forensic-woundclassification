# Ticket T17: Per-Box ROI Experiment (no union)

- **Ticket ID**: `T17`
- **Phase**: Extra experiment, outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §1 (main question); [T03](T03_cvat_roi_cropping.md) (crops); [T08b](T08b_lr_tuning.md) (tuning); [T15](T15_summary_and_full_vs_roi.md) (paired comparison)
- **Scenarios**: S17–S24 (input path `box`, the 8 families in S01–S16 order)
- **Status**: Implemented; HPC run pending

---

## 1. Goal

The main study combines all boxes of a photo into one padded union crop. This experiment crops **every box on its own** and asks whether per-box crops classify better than the union crop and the full image, on the same 157 test photos from 62 cases. It changes nothing in the main run: separate manifest, crop folder and experiments root.

---

## 2. Decisions (2026-10-15, before any per-box model was trained)

| Item | Decision | Reason |
|---|---|---|
| Crops | `create_roi_dataset.py --crop-mode individual`, 8% padding, min size 8 px: one crop per box, 1,667 crops (train 1,177 / validation 277 / test 213) from the 1,100 annotated photos, 418 cases | Same settings as the union crops; only the grouping differs |
| Label | A crop takes its photo's label | Each CVAT task holds one wound class, so every box of a photo was drawn under that photo's class |
| Split | Each crop inherits its photo's case-level split; leakage check on cases | Same frozen split as the main study |
| Photos with many boxes (max 35) | Training loss weight per crop = 1 / (train crops of its photo), so each photo counts once; class weights from these weighted counts (= photos per class, the same as the union path) | User decision: weight per photo |
| Photo-level prediction | Mean of the crop probabilities of a photo; predicted class = argmax | User decision: image level is primary |
| Model selection | Early stopping and learning-rate choice on **photo-level** validation macro-F1 (ties: lower crop-level validation loss) | Selection uses the primary unit |
| Families and budget | All 8 families, same rules as S01–S16: lr grid {1e-5, 3e-5, 1e-4} for S17–S22, CNN-SVM grid (SVM fitted with the photo weights), AlexNet ensemble seeds 43–45; max 200 epochs, patience 25 | User decision: all families; comparable with the main study |
| Primary outcome | Test **photo-level** macro-F1, box − union (main `roi`) per family, paired case-level bootstrap (2,000 resamples, seed 42), Holm over 8 families, pooled mean | Same method as T15 |
| Secondary | Box − full image; balanced accuracy and macro AUC; crop-level test metrics (213 crops) reported descriptively | |
| Not done | Grad-CAM, cross-validation, final verification (T16) for S17–S24 | Not requested; can be added later |

---

## 3. Requirements

1. **Crops**: committed dry-run manifest `data/roi_box/roi_manifest.csv` (+ `crop_summary.json`); live crops on Raptor in `derived_data/roi_box_p008`, checked against the manifest by `compare_roi_manifests.py` (`t03_roi_crops.sbatch` with `CROP_MODE=individual`).
2. **Training** (`train.py` / `cnn_svm.py --input-path box`): images from `roi_relative_path`; photo weights in the loss (SVM: `sample_weight`); photo-level validation macro-F1 for selection; prediction files keep one row per crop and add `source_relative_path`.
3. **Ensemble**: soft vote per crop; `source_relative_path` is carried over.
4. **Crop-level evaluation**: `evaluate.py --experiments-root <exp>/roi_box` evaluates `box/<family>/` as S17–S24.
5. **Photo level and comparison** (`src/evaluation/box_summary.py --box-root <exp>/roi_box --main-root <exp>/main`):
   - per scenario `box/<family>/photo/`: photo-level prediction files and the usual metric files;
   - `summary/box_summary_table.csv`: per family photo-level and crop-level test metrics;
   - `summary/box_vs_union.csv`, `summary/box_vs_full.csv`: paired comparisons (columns as T15 with `reference`/`box`);
   - `summary/box_vs_union_macro_f1.png`: forest plot.
   - Every photo-level scenario must cover the same test cases as the main run.
6. **HPC**: `train_single.sbatch` accepts `INPUT_PATH=box`; `submit_all.sh` runs any set of paths via `PATHS` (job names S17–S24 for `box`).
7. **Tests** (`tests/test_box_roi.py`, CPU, synthetic): photo weights and class weights; photo mean; training on `box` writes crop rows with `source_relative_path` and selects on photo level; CNN-SVM on `box`; ensemble keeps the column; evaluate names S17–S24 and leaves the main index unchanged; `box_summary.py` end to end.

---

## 4. Files

| File | Change |
|---|---|
| `src/training/photo.py` | New: photo weights, photo mean, photo-level prediction frame |
| `src/training/config.py`, `data.py`, `engine.py`, `train.py`, `common.py`, `cnn_svm.py`, `ensemble.py` | `box` input path, weighted loss, photo-level selection |
| `src/evaluation/evaluate.py` | `box` scenarios S17–S24 |
| `src/evaluation/box_summary.py`, `plots.py` | New summary; forest-plot labels |
| `scripts/slurm/train_single.sbatch`, `t03_roi_crops.sbatch`, `scripts/hpc/submit_all.sh` | `box` path, crop mode, `PATHS` |
| `data/roi_box/` | Dry-run manifest |
| `tests/test_box_roi.py` | Tests |
| `runbook.md`, `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Commands, scope, memo |

---

## 5. Acceptance Check

1. `pytest tests/` passes (locally and in the Raptor container).
2. Live crops match the committed manifest (1,667 crops).
3. 8 of 8 scenarios finish; `evaluate.py` evaluates S17–S24 on 213 test crops from 62 cases; `box_summary.py` writes the four files on 157 photos and 62 cases.
4. Record the results here.

---

## 6. HPC Run

Pending.
