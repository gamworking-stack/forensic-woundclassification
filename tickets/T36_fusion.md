# Ticket T36: Soft Voting between the Full-Image and Crop Models

- **Ticket ID**: `T36`
- **Phase**: Extra experiment on the T23 runs (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T23](T23_relabel_rerun.md) (runs), [T25](T25_patient_cases.md) (patients as cases), [T30](T30_more_contrast.md) (paired comparison), [T35](T35_quality_accuracy.md)
- **Scenarios**: none new; the T23 validation and test predictions of S01–S15 (full image) and S25–S32 (per-crop), combined
- **Status**: Done (2026-11-15; results §5)

---

## 1. Goal

The researcher asked (2026-11-15) for a pipeline that raises accuracy by letting the full-image and crop models vote.

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Rule | Soft voting: weighted mean of the class probabilities of the full-image and per-crop model of the same family | Two voters cannot hold a majority vote; probabilities keep each model's confidence |
| Photo level | A photograph: `w × full + (1 − w) × mean of its crops`; scored on the 149 test photographs against the full-image model (Table 2) | The full image's own unit |
| Crop level | A crop: `w × full image of its photograph + (1 − w) × crop`; scored on the 276 test crops against the per-crop model (Table 2) | The per-crop unit |
| Weight | `equal`: w = 0.5, fixed before any result (primary). `tuned`: w in 0, 0.1, …, 1 with the best validation macro-F1, per family and level, ties to the weight nearest 0.5 (secondary) | The test split is never used to choose. Validation also chose each model's learning rate and epoch, so the tuned weight is slightly optimistic for validation, not for test |
| Families | All 8, each fused with itself only | No choice among model pairs, so nothing is selected on test |
| Statistics | Fused minus original on the same test images: `patient_rescore.py` then `contrast_compare.py` (paired patient bootstrap, 2,000 resamples, Holm over the 8 families, per class) | Same method as T30 |
| Matching | A crop to its photograph by group and file stem (as `patient_rescore.py`); every photograph must have crops and every crop a photograph | Full-image rows name the raw photo file |

## 3. Files

| File | Change |
|---|---|
| `src/evaluation/fusion.py` | New |
| `tests/test_fusion.py` | Equal-weight averages at both levels, rows kept in order, tuned weight follows the better model on validation, fused files score with `contrast_compare.py`, refusals |
| `scripts/slurm/fusion.sbatch` | New CPU job: fuse, copy originals, rescore per patient, compare |

## 4. Run log

- 2026-11-15 (laptop): `pytest tests/test_fusion.py tests/test_contrast_compare.py tests/test_quality_accuracy.py` 14 passed (Python 3.11, CPU, no torch).
- 2026-11-15 (Raptor, commit `5e082a0`): job 247175 submitted.
- Job 247175 completed in 21 min: tests passed; 32 fused files per rule; all files rescored to 391 patients; both comparisons written. Outputs: `$EXPERIMENTS_DIR/fusion/`; summaries and figures (no per-image rows) copied to `../results_figures/Fusion full and crop/compare_{equal,tuned}/`.

## 5. Results (test: 149 photographs / 276 crops, 55 patients)

Equal weights (primary), fused minus original, mean of the 8 families [95% CI]:

| Level | Accuracy | Macro-F1 |
|---|---|---|
| Photo (vs full image) | 0.791 → 0.862, **+0.071 [+0.041, +0.105]** | 0.683 → 0.777, **+0.094 [+0.038, +0.145]** |
| Crop (vs per-crop) | 0.806 → 0.846, **+0.040 [+0.018, +0.066]** | 0.705 → 0.745, **+0.040 [+0.005, +0.100]** |

Accuracy per family, original → fused (equal); * = Holm p < 0.05:

| Family | Photo level | Crop level |
|---|---|---|
| EfficientNet-B0 | 0.866 → 0.886 | 0.822 → 0.877* |
| ResNet152 | 0.893 → 0.893 | 0.819 → 0.880* |
| MobileNetV3-Small | 0.745 → 0.826 | 0.790 → 0.804 |
| GoogLeNet | 0.826 → 0.899* | 0.819 → 0.866 |
| VGG16 | 0.819 → 0.893* | 0.833 → 0.884* |
| ViT-B/16 | 0.792 → 0.872* | 0.830 → 0.866 |
| CNN-SVM | 0.651 → 0.812* | 0.819 → 0.808 |
| AlexNet ensemble | 0.738 → 0.819* | 0.717 → 0.786* |

- **Voting helps clearly, at both levels.** Every pooled interval excludes 0. Photo level: 7 of 8 families gain, 6 significantly after Holm; ResNet152, the best full-image model, is unchanged (0.893). Crop level: 6 of 8 gain, 4 significantly; CNN-SVM and MobileNetV3-Small barely move.
- **The best single result moves little**: best photo-level accuracy 0.899 (GoogLeNet fused) against 0.893 (ResNet152 alone); best crop-level 0.884 (VGG16 fused) against 0.833 (VGG16 alone). Voting mostly lifts the weaker models to the level of the best.
- **Per class (pooled recall, equal)**: photo level graze +0.066 [+0.034, +0.105] and laceration +0.102 [+0.011, +0.196]; crop level gunshot +0.054 [+0.027, +0.083] and laceration +0.120 [+0.035, +0.165]. Scratch (photo +0.229) and penetration change too, on 6 and 3 patients.
- **Tuned weights** (weight on the full image 0.2–0.6 at photo level, 0.0–0.6 at crop level) give about the same: photo accuracy +0.080 [+0.040, +0.123], macro-F1 +0.106 [+0.035, +0.162]; crop +0.039 [+0.023, +0.057], +0.038 [+0.016, +0.079]. Tuning adds nothing over 0.5, so equal weights are the rule to report.
- **Caveat for use**: the photo-level vote needs the wound boxes (here the manual CVAT boxes) as well as the photograph; the crop-level vote needs the whole photograph as well as the box. A fully automatic pipeline needs a wound detector (Experiment 4).
