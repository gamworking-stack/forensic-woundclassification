# Ticket T17: Per-Box ROI Experiment (no union)

- **Ticket ID**: `T17`
- **Phase**: Extra experiment, outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §1 (main question); [T03](T03_cvat_roi_cropping.md) (crops); [T08b](T08b_lr_tuning.md) (tuning); [T15](T15_summary_and_full_vs_roi.md) (paired comparison)
- **Scenarios**: S17–S24 (input path `box`, the 8 families in S01–S16 order)
- **Status**: Completed (2026-10-15)

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

## 6. HPC Run (2026-10-15)

- **Crops**: job 229660 (`t03_roi_crops.sbatch`, `CROP_MODE=individual`), 6 min: 1,667 crops in `derived_data/roi_box_p008`, live crops match the committed manifest; `tests/test_roi_preprocessing.py` 23 passed.
- **First training attempt** (jobs 229661–229689, commit `7f50d4c`): all 22 training jobs failed at the first validation epoch and the 7 dependent jobs were cancelled. The committed dry-run manifest leaves `source_relative_path` empty, so every crop fell into one photo. Fixed in `54d1812` (photo = manifest `relative_path`; 1,100 photos, one label, split and case each); no checkpoint had been written, so nothing was reused.
- **Training** (jobs 229690–229718, commit `54d1812`): 29 of 29 completed on the first attempt; 2–12 min per grid run.
- **Evaluation**: `evaluate.py` (crop level, 213 test crops, 62 cases) and `box_summary.py` (job 229725, 6 min; 157 test photos, 62 cases, the same cases as the main run).

**Choices on validation (photo-level macro-F1)**

| Scenario | Family | Chosen | Best epoch | Validation macro-F1 |
|---|---|---|---|---|
| S17 | EfficientNet-B0 | lr 1e-5 | 54 | 0.798 |
| S18 | ResNet152 | lr 1e-4 | 19 | 0.832 |
| S19 | MobileNetV3-Small | lr 3e-5 | 31 | 0.801 |
| S20 | GoogLeNet | lr 1e-4 | 7 | 0.762 |
| S21 | VGG16 | lr 3e-5 | 9 | 0.777 |
| S22 | ViT-B/16 | lr 3e-5 | 2 | 0.801 |
| S23 | CNN-SVM | RBF, C 1000, gamma 1e-6 | – | 0.718 |
| S24 | AlexNet ensemble | seeds 43–45 (lr 3e-5) | 17 / 10 / 22 | 0.761 / 0.740 / 0.737 |

Grid edges: S17 and S20 chose a grid limit (1e-5, 1e-4), S18 the upper limit; S23 sits on both the largest C and the smallest gamma, as in the main run (T06). The grids were not widened.

**Test, photo level (primary), 157 photos, 62 cases**

| Scenario | Family | Accuracy | Balanced acc. | Macro-F1 [95% CI] | Macro AUC | Crop-level macro-F1 (213 crops) |
|---|---|---|---|---|---|---|
| S17 | EfficientNet-B0 | 0.879 | 0.812 | **0.811** [0.703, 0.878] | 0.969 | 0.755 |
| S18 | ResNet152 | 0.860 | 0.794 | 0.763 [0.672, 0.829] | 0.914 | 0.712 |
| S19 | MobileNetV3-Small | 0.834 | 0.809 | 0.777 [0.663, 0.846] | 0.973 | 0.720 |
| S20 | GoogLeNet | 0.841 | 0.791 | 0.747 [0.644, 0.828] | 0.975 | 0.673 |
| S21 | VGG16 | 0.854 | 0.782 | 0.792 [0.683, 0.867] | 0.973 | 0.733 |
| S22 | ViT-B/16 | 0.847 | 0.837 | 0.779 [0.693, 0.836] | 0.975 | 0.723 |
| S23 | CNN-SVM | 0.828 | 0.728 | 0.734 [0.612, 0.820] | 0.972 | 0.702 |
| S24 | AlexNet ensemble | 0.854 | 0.762 | 0.752 [0.635, 0.825] | 0.967 | 0.705 |

**Paired comparison, test macro-F1 (photo level)**

| Family | Union ROI | Full | Per-box | Box − union [95% CI], Holm p | Box − full [95% CI], Holm p |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0.742 | 0.733 | 0.811 | +0.069 [+0.025, +0.137], 0.016 | +0.078 [−0.063, +0.212], 1.000 |
| ResNet152 | 0.726 | 0.783 | 0.763 | +0.037 [−0.068, +0.142], 1.000 | −0.020 [−0.126, +0.083], 1.000 |
| MobileNetV3-Small | 0.727 | 0.653 | 0.777 | +0.050 [−0.010, +0.115], 0.889 | +0.124 [+0.000, +0.216], 0.350 |
| GoogLeNet | 0.736 | 0.705 | 0.747 | +0.012 [−0.064, +0.116], 1.000 | +0.042 [−0.074, +0.178], 1.000 |
| VGG16 | 0.715 | 0.699 | 0.792 | +0.077 [−0.043, +0.202], 1.000 | +0.093 [−0.016, +0.202], 0.576 |
| ViT-B/16 | 0.761 | 0.754 | 0.779 | +0.018 [−0.060, +0.093], 1.000 | +0.025 [−0.090, +0.138], 1.000 |
| CNN-SVM | 0.725 | 0.585 | 0.734 | +0.009 [−0.092, +0.100], 1.000 | +0.148 [+0.011, +0.271], 0.264 |
| AlexNet ensemble | 0.728 | 0.662 | 0.752 | +0.024 [−0.034, +0.082], 1.000 | +0.091 [−0.017, +0.181], 0.576 |
| **All families (mean)** | 0.732 | 0.697 | 0.769 | **+0.037 [+0.006, +0.071]**, p = 0.027 | **+0.073 [+0.009, +0.129]**, p = 0.028 |

Secondary (pooled, box − union): balanced accuracy +0.040 [+0.012, +0.071], p = 0.011 (EfficientNet-B0 Holm p < 0.001); macro AUC +0.009 [−0.002, +0.022], p = 0.101. Box − full: balanced accuracy +0.080 [−0.001, +0.155], p = 0.053; macro AUC +0.030 [+0.003, +0.063], p = 0.027.

**Reading**
- Per-box is higher than union ROI in 8 of 8 families and than the full image in 7 of 8 (ResNet152 lower). The pooled differences exclude 0 for both comparisons; per family only EfficientNet-B0 vs union survives Holm. S17 EfficientNet-B0 (0.811) is the highest test macro-F1 of all 24 scenarios.
- Only 30 of the 157 test photos have more than one box; for the other 127 the per-box crop equals the union crop. The gain therefore comes partly from scoring multi-box photos as the mean of their wounds and partly from models trained on individual wound crops (1,177 training crops instead of 779 union crops). This run cannot separate the two.
- Crop-level scores (0.67–0.76) are lower than photo-level: a single crop is harder than the photo, and averaging a photo's crops helps.
- Limits: the same 62 test cases as the main study (third use of this test set, for a new input path; no setting was chosen on it); post hoc relative to the thesis proposal; one training seed per setting; per-box crops still need manual boxes.

---

## 7. Supplementary Analyses (decided 2026-10-15, after the §6 test results were seen)

Requested to bring the per-box run to the same reporting level as the main study. They change no model or test result and are reported as post hoc.

| Item | Decision |
|---|---|
| Cross-validation | T12b method on development data: the frozen `cv_folds.csv`, each S17–S24 scenario's §6 settings (chosen lr and best epoch as a fixed epoch count, chosen SVM, member seeds and epochs), held-out fold predicted per crop and scored on photo level; paired comparison of out-of-fold photos with the T12b union ROI and full-image out-of-fold predictions (same 356 cases, 943 photos), 2,000 resamples, Holm over 8 families. Code: `cv_plan.py --paths box`, `submit_cv.sh` (`PATHS`), `src/evaluation/cv_box_summary.py` |
| Grad-CAM | T14 method on every crop of the main run's 14 example photos (same photos, read from `main/gradcam/examples.csv`), the 7 families of T14 (CNN-SVM excluded), the checkpoint behind each S17–S24 test file; crop and photo predictions checked against the prediction files. No "map inside the box" score: a per-box crop is the box. Code: `gradcam.py --box-manifest --examples` |
| Verification | T16 checks for S17–S24 (`src/evaluation/verify_box.py`): files, test set equal to the main run's, selection reproduced on photo level, budget equal to the main run, every number recomputed, Grad-CAM and CV. `verify.py` now scores per-box histories on photo level (no change for the main run) |
| Reporting | Metrics by wound type (Excel) and slides, from the fetched result files |

