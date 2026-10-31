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
