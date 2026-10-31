# Ticket T30: "More Contrast" Experiment (CLAHE-Enhanced Images, Retrained)

- **Ticket ID**: `T30`
- **Phase**: Extra experiment on the T23 dataset (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T23](T23_relabel_rerun.md) (data, split, runs), [T25](T25_patient_cases.md) (patient intervals), [T28](T28_precision_recall.md), [T29](T29_confidence_threshold.md)
- **Scenarios**: S01–S15 (full image) and S25–S32 (per-crop) retrained and tested on contrast-enhanced images; the T23 runs are not touched
- **Status**: Done (2026-10-31; results §5)

---

## 1. Goal

The researcher asked (2026-10-31) to add more contrast to every image and rerun the experiment, named "more contrast experiment". Decisions (researcher, 2026-10-31): retrain (not test-only), CLAHE, full image and per-crop, reported with the same metrics and the 0.60 confidence threshold (T29).

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Enhancement | CLAHE on CIELAB L* only (OpenCV 4.10, clip limit 2.0, 8 × 8 tiles); a* and b* unchanged | Researcher's choice; local contrast without changing wound colour |
| Where | Once per photograph, after the EXIF orientation, before anything else; saved as JPEG (quality 95, no chroma subsampling, no EXIF) at full size in `$DERIVED_DIR/raw_clahe` (`src/data/create_contrast_dataset.py`) | Every reader sees the same upright image at the same size |
| Crops | Cut from the enhanced photographs by the T03 crop job with the T23 per-box boxes and padding, checked against the committed T23 `roi_box/roi_manifest.csv` | Same crops, only the pixels differ |
| Data, split, boxes | T23 (`WOUND_DATA_DIR=data/relabel_2026_10_23`), unchanged | Only contrast differs from T23 |
| Training | Identical to T23: same families, learning-rate grid and selection on validation, seeds, augmentation (which still includes ±0.2 contrast jitter), epochs | One change at a time |
| Runs | `experiments_relabel_2026_10_23_more_contrast/{main,roi_crop}`, crops in `derived_data/relabel_2026_10_23_more_contrast/`; `scripts/hpc/contrast_env.sh` sets every folder | Own folders, as T23 |
| Code change | `train_single.sbatch` reads `RAW_IMAGE_ROOT` (default: the raw photographs) | Full-image runs need another photo folder |
| Reporting | Table 2 metrics with patient-level intervals (T25, T28), the 0.60 threshold (T29), and CLAHE minus T23 per scenario on the same test images | Same metrics as before, plus the comparison asked for |

## 3. Files

| File | Change |
|---|---|
| `src/data/create_contrast_dataset.py` | New: CLAHE copy of an image tree, `contrast_manifest.csv`, `contrast_report.json` |
| `tests/test_contrast_dataset.py` | Contrast up and colour kept; tree mirrored upright without EXIF and found by `find_source_image`; output inside the source refused |
| `scripts/slurm/contrast_images.sbatch` | New CPU job |
| `scripts/hpc/contrast_env.sh` | New |
| `scripts/slurm/train_single.sbatch` | `RAW_IMAGE_ROOT` |
| `runbook.md` (T23 step 13), `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Commands, scope, memo |

## 4. Run log

- 2026-10-31: `pytest tests/test_contrast_dataset.py tests/test_contrast_compare.py` 5 passed (laptop, Python 3.11, OpenCV 4.10).
- Raptor, commit `c9c6faf`: job 241622 failed on a macOS `._PB106186.JPG` metadata file (2 in the raw tree, never read by any run); fixed by skipping hidden files. Job 241623 enhanced the 1,110 photos (47 s). Check on 40 random photos (statistics only, no image left Raptor): sizes kept 40/40; local contrast (9 × 9 window SD of L*) median ×1.45 at 224 px (10–90%: ×1.32–1.62), edge strength ×1.55; global L* SD ×1.00; mean a*/b* change 0.4 of 255.
- Crop job 241627: 1,808 per-box crops, `ROI comparison PASSED` against the T23 manifest.
- Training jobs 241628–241685 (58): all completed 17:05–19:00. `run_config.json`: `image_root` the CLAHE folders, `git_commit` c9c6faf, `git_dirty: true` only because of the untracked T19 files on Raptor (as in T23).
- Evaluation: `evaluate.py` and `crop_summary.py` on Raptor (job 241696, group-level). Laptop (runbook step 13): the 32 T30 and 16 T23 prediction files copied to a temporary folder, rewritten per patient (`patient_rescore.py`), scored (`evaluate.py`, `crop_summary.py`, `precision_recall.py`, `confidence_threshold.py`, `contrast_compare.py`), deleted. Outputs: `../results_figures/More contrast experiment/`. `confidence_threshold.py` reproduced every new `test_metrics.json` at threshold 0; the original side of `contrast_vs_original.csv` equals Table 2.

## 5. Results (test: 149 photos / 276 crops, 55 patients)

CLAHE minus original (T23) on the same test images; 95% patient-level paired bootstrap, mean of the 8 families:

| Input | Accuracy | Balanced accuracy | Macro-F1 | Macro AUC |
|---|---|---|---|---|
| Full image | 0.791 → 0.779, −0.012 [−0.029, +0.005] | −0.021 [−0.062, +0.001] | 0.683 → 0.661, −0.021 [−0.055, +0.003] | −0.006 [−0.014, +0.005] |
| Per-crop | 0.806 → 0.820, +0.014 [−0.003, +0.031] | +0.022 [−0.034, +0.055] | 0.705 → 0.726, +0.021 [−0.027, +0.057] | +0.010 [−0.004, +0.020] |

- **No clear effect.** Every pooled interval includes 0. Full image leans slightly worse, per-crop slightly better.
- Per model, only the per-crop AlexNet ensemble changes after Holm adjustment (accuracy 0.717 → 0.768, +0.051 [+0.018, +0.087], Holm p 0.04). Full-image AlexNet ensemble macro-F1 −0.084 [−0.162, −0.002] has nominal p 0.04, Holm p 0.34. Best full-image model is still ResNet152 (accuracy 0.893, as in T23). Best per-crop accuracy is now EfficientNet-B0, 0.855 (its T23 value 0.822, +0.033 [−0.004, +0.069]; T23's best per-crop was VGG16, 0.833).
- With the 0.60 threshold (T29) the picture is the same: full image 22.5% inconclusive, accuracy on the rest 0.861 [0.803, 0.911] (T23: 23.6%, 0.866); per-crop 16.3%, 0.868 [0.775, 0.936] (T23: 16.9%, 0.866).
- The training augmentation already varies contrast by ±20%, and CLAHE leaves global contrast and colour unchanged, so a small effect is plausible. One clip limit and tile size were tried, chosen before any result; no tuning.

## 6. Per wound class (added 2026-10-31)

The researcher asked for the added value of CLAHE per wound class. `contrast_compare.py` now also scores recall, precision, F1 and one-vs-rest AUC per class on the same paired patient resamples (pooled rows Holm-adjusted over the 7 classes), and counts per model the images gained (original wrong, CLAHE right) and lost. Rerun on fresh prediction copies (deleted after); the overall table was reproduced exactly. Outputs: `contrast_per_class.csv`, `contrast_per_class_changes.csv`, `contrast_per_class_{recall,f1}.png`.

Recall (= accuracy within the class), mean of 8 families, CLAHE minus original [95% CI]; mean images gained / lost per model:

| Class | Full image: test photos (patients) | Recall change | Gained / lost | Per-crop: test crops (patients) | Recall change | Gained / lost |
|---|---|---|---|---|---|---|
| Gunshot wound | 41 (16) | +0.006 [−0.018, +0.036] | 1.4 / 1.1 | 77 (16) | **+0.036 [+0.006, +0.058]** | 4.6 / 1.9 |
| Graze wound | 40 (20) | −0.006 [−0.040, +0.026] | 1.8 / 2.0 | 83 (20) | −0.020 [−0.050, +0.010] | 3.5 / 5.1 |
| Contusion | 23 (14) | −0.043 [−0.104, +0.000] | 0.8 / 1.8 | 24 (14) | +0.010 [−0.010, +0.047] | 0.6 / 0.4 |
| Laceration | 16 (12) | +0.023 [−0.025, +0.089] | 1.0 / 0.6 | 24 (12) | **+0.062 [+0.008, +0.112]** | 2.1 / 0.6 |
| Cut wound | 13 (5) | **−0.067 [−0.125, −0.047]** | 0.4 / 1.3 | 31 (5) | +0.008 [−0.025, +0.062] | 2.1 / 1.9 |
| Scratch | 12 (6) | +0.000 [−0.172, +0.094] | 1.0 / 1.0 | 29 (6) | +0.013 [−0.042, +0.028] | 2.0 / 1.6 |
| Penetration wound | 4 (3) | −0.062 [−0.250, +0.000] | 0.3 / 0.5 | 8 (3) | +0.047 [−0.312, +0.250] | 1.0 / 0.6 |

- **Per-crop, where CLAHE helps most:** gunshot wounds (recall +0.036, 6 of 8 models up; net +2.8 crops of 77 per model) and lacerations (+0.062, 5 of 8 up; net +1.5 of 24). Precision also rises for contusion (+0.056 [+0.007, +0.114]), cut wound (+0.067 [+0.000, +0.134]) and graze wound (+0.024 [+0.003, +0.059]); F1 for contusion +0.038 [+0.005, +0.083] and cut wound +0.024 [+0.005, +0.090]; cut-wound AUC +0.037 [+0.017, +0.058] (the only per-class change on per-crop with Holm p < 0.05, 0.035). Graze-wound recall leans down (−0.020; 5 of 8 down, net −1.6 of 83).
- **Full image, no class gains:** cut-wound recall −0.067 (0.683 → 0.615; Holm p < 0.001) and F1 −0.064 [−0.105, −0.032] (Holm p 0.014); contusion recall −0.043 [−0.104, +0.000] and AUC −0.019 [−0.043, −0.002]. Every other class is unchanged.
- **Caution:** cut wounds come from 5 test patients and penetration wounds from 3; with so few clusters the patient bootstrap understates the uncertainty (as for the living subgroup, T24 §7), so the narrow cut-wound intervals are not reliable evidence. The full-image cut-wound loss is about 1 photo of 13 per model (8.9 → 8.0 correct). Per family, 6 of 112 recall comparisons have p < 0.05, about the number expected by chance.
- **Reading:** on crops, enhanced local contrast slightly helps the classes defined by edges and texture (gunshot entry margins, laceration edges; precision for contusion and cut wound), with nothing lost elsewhere except a small graze-wound dip. On full images, where the wound is a small part of the picture, CLAHE also sharpens surrounding skin and background and gives no class a gain. Post hoc and descriptive.

## 7. Five models per wound class (added 2026-10-31)

The researcher asked to focus on ResNet152, EfficientNet-B0, VGG16, ViT-B/16 and MobileNetV3-Small, per wound class. `contrast_compare.py --families ...` compares and pools only those (pooled row "mean of 5 families", Holm over the 5); each model's own values are identical to the 8-model run (checked, 280 rows), since the resamples do not depend on the families. Outputs: `../results_figures/More contrast experiment/five_models/` (same files as §5–6, plus `contrast_per_class_{recall,f1}_by_model.png`, model × class heatmaps).

Overall (mean of 5): full image accuracy 0.823 → 0.813, −0.009 [−0.039, +0.020]; per-crop 0.819 → 0.836, +0.017 [−0.003, +0.037].

Recall, original → CLAHE (images gained / lost); * = 95% interval excludes 0:

| Per-crop | EfficientNet-B0 | ResNet152 | MobileNetV3-S | VGG16 | ViT-B/16 | Mean of 5 [95% CI] |
|---|---|---|---|---|---|---|
| Gunshot (77) | 0.95 → 0.95 (2/2) | 0.88 → 0.94 (4/0) | 0.86 → 0.95 (8/1)* | 0.88 → 0.96 (7/1)* | 0.94 → 0.96 (3/1) | **+0.049 [+0.023, +0.072]** |
| Laceration (24) | 0.71 → 0.79 (2/0) | 0.75 → 0.75 (1/1) | 0.62 → 0.83 (5/0) | 0.67 → 0.75 (3/1) | 0.71 → 0.75 (1/0) | +0.083 [+0.000, +0.153] |
| Graze (83) | 0.87 → 0.93 (7/2)* | 0.90 → 0.90 (5/5) | 0.90 → 0.87 (3/6) | 0.96 → 0.84 (0/10)* | 0.90 → 0.89 (4/5) | −0.022 [−0.049, +0.010] |
| Scratch (29) | 0.79 → 0.90 (5/2) | 0.79 → 0.76 (1/2) | 0.83 → 0.83 (1/1) | 0.76 → 0.83 (2/0) | 0.90 → 0.83 (0/2) | +0.014 [−0.160, +0.043] |
| Cut (31) | 0.48 → 0.48 (1/1) | 0.52 → 0.55 (3/2) | 0.45 → 0.55 (3/0) | 0.65 → 0.52 (2/6) | 0.52 → 0.52 (4/4) | +0.000 [−0.040, +0.100] |
| Contusion (24) | 0.92 → 0.88 (0/1) | 0.92 → 0.92 (1/1) | 0.83 → 0.83 (0/0) | 0.92 → 0.92 (1/1) | 0.83 → 0.88 (1/0) | +0.000 [−0.035, +0.050] |
| Penetration (8) | 0.62 → 0.62 (1/1) | 0.50 → 0.62 (1/0) | 0.50 → 0.38 (0/1) | 0.25 → 0.38 (2/1) | 0.38 → 0.38 (1/1) | +0.025 [−0.400, +0.250] |

| Full image | EfficientNet-B0 | ResNet152 | MobileNetV3-S | VGG16 | ViT-B/16 | Mean of 5 [95% CI] |
|---|---|---|---|---|---|---|
| Gunshot (41) | 0.98 → 0.90 (0/3) | 1.00 → 0.98 (0/1) | 0.90 → 0.98 (3/0) | 0.90 → 0.95 (2/0) | 0.93 → 0.90 (0/1) | +0.000 [−0.036, +0.036] |
| Graze (40) | 0.90 → 0.85 (2/4) | 0.93 → 0.97 (2/0) | 0.75 → 0.70 (2/4) | 0.85 → 0.93 (3/0) | 0.90 → 0.90 (2/2) | +0.005 [−0.055, +0.060] |
| Contusion (23) | 0.91 → 0.87 (1/2) | 0.87 → 0.87 (1/1) | 0.96 → 0.70 (0/6)* | 0.83 → 0.83 (1/1) | 0.74 → 0.74 (1/1) | −0.061 [−0.140, +0.000] |
| Laceration (16) | 0.75 → 0.75 (0/0) | 0.94 → 0.88 (0/1) | 0.69 → 0.62 (0/1) | 0.75 → 0.94 (3/0) | 0.56 → 0.62 (1/0) | +0.025 [−0.031, +0.100] |
| Cut (13) | 0.85 → 0.69 (0/2) | 0.77 → 0.77 (0/0) | 0.69 → 0.54 (1/3) | 0.69 → 0.46 (0/3) | 0.85 → 0.85 (0/0) | −0.108 [−0.267, −0.040] |
| Scratch (12) | 0.58 → 0.58 (0/0) | 0.67 → 0.67 (0/0) | 0.08 → 0.42 (4/0) | 0.67 → 0.67 (1/1) | 0.58 → 0.67 (2/1) | +0.083 [−0.120, +0.200] |
| Penetration (4) | 0.50 → 0.75 (1/0) | 0.50 → 0.50 (0/0) | 0.25 → 0.25 (0/0) | 0.75 → 0.25 (0/2) | 0.00 → 0.00 (0/0) | −0.050 [−0.200, +0.000] |

- **Gunshot wounds, per-crop: the one consistent gain.** All 5 models are equal or better (net +3.8 crops of 77 per model; mean +0.049, Holm p < 0.001), largest for MobileNetV3-Small and VGG16 (+0.09, +0.08).
- **Lacerations, per-crop:** 4 of 5 better, none worse (mean +0.083, interval touching 0); MobileNetV3-Small +0.21 (5 crops gained, 0 lost).
- **Graze wounds, per-crop: model-dependent.** EfficientNet-B0 gains (+0.06*, 7/2) but VGG16 loses 10 crops and gains none (−0.12*).
- **Full image: no consistent gain for any class.** Cut wounds drop in 3 of 5 models (mean −0.108; 5 test patients, so the interval is too narrow to trust); MobileNetV3-Small loses 6 of 23 contusions (0.96 → 0.70*) but gains 4 of 12 scratches (0.08 → 0.42).
- **Per model:** MobileNetV3-Small, the weakest model, changes most in both directions; ResNet152, the strongest on full images, barely changes (full-image accuracy 0.893 both times). EfficientNet-B0 becomes the best per-crop model (0.855) but drops on full images (0.866 → 0.819).
- F1 (heatmap): per-crop contusion +0.05* (EfficientNet-B0 +0.15*); full-image laceration VGG16 +0.16*. Post hoc and descriptive; with 5 models × 7 classes × 2 inputs some intervals exclude 0 by chance.
