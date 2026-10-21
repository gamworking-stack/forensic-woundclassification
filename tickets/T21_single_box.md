# Ticket T21: Single-Box Test Photos, One Full Image vs One Crop

- **Ticket ID**: `T21` (`T20` is reserved for the Experiment 5 plan)
- **Phase**: Extra analysis, outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T15](T15_summary_and_full_vs_roi.md) (paired case-level bootstrap); [T17](T17_box_roi.md) (the per-box crops); [T18](T18_crop_roi.md) (the per-crop models)
- **Scenarios**: none new; reuses the test predictions of S01–S16 (main run) and S25–S32 (`roi_crop`)
- **Status**: Completed (2026-10-21)

---

## 1. Goal

Show, one input against one input, whether a crop of the wound is classified better than the full photograph. On a test photo with **exactly one annotated box**, the photo, its per-box crop and its union crop show the same single wound. A full-image model and a crop model then each make one prediction from one image, and nothing is averaged over crops. This removes the two things that blur the Experiment 2 and 3 readings: averaging several crops per photo, and photos with many wounds counting once per crop.

No model is trained and no new test prediction is made: the analysis reads the existing test prediction files.

---

## 2. Decisions (2026-10-21, after every test result had been seen)

Decided with the researcher after the test results of Experiments 1–3, so the thesis reports T21 as a **post hoc, exploratory** analysis.

| Item | Decision | Reason |
|---|---|---|
| Photos | Test photos with exactly one box in `data/roi_box/roi_manifest.csv`: **127 photos from 57 cases** (of 157 / 62) | One wound, one crop, one prediction per input |
| Per class (photos / cases) | contusion 20 / 11, cut wound 10 / 7, graze wound 36 / 20, gunshot wound 32 / 14, laceration 18 / 12, penetration wound 7 / 5, scratch 4 / 4 | Fact of the subset |
| Primary comparison | Per-crop ROI (S25–S32) − full image (S01, S03, …, S15), test macro-F1 per family on the 127 photos | The Experiment 3 models, each scored on one crop |
| Secondary comparison | Union ROI (S02, …, S16) − full image on the same photos | The union crop of a one-box photo is that box's crop, and these models were trained on the same photos as the full-image models |
| Statistics | T15 paired case-level bootstrap: 2,000 resamples of the 57 cases, seed 42, 95% percentile interval, two-sided p, Holm over the 8 families, pooled mean over families | Same method as every other comparison |
| Metrics | Macro-F1 (primary), balanced accuracy, macro AUC with intervals; accuracy as a point value | T15 metric set, plus accuracy for the presentation |
| Matching | Per-crop rows by crop file (`roi_relative_path`), union rows by union crop file, full-image rows by case and file stem; every photo must be found exactly once, with its own case and class | Prediction files of the `crop` path name only the crop |

---

## 3. Requirements

1. `src/evaluation/single_box.py --main-root <exp>/main --crop-root <exp>/roi_crop` writes
   - the single-box test predictions to `roi_crop/single_box/<full|roi|crop>/<family>/test_predictions.csv` (per-image files stay on Raptor);
   - `roi_crop/summary/single_box_counts.csv` (photos and cases per class), `single_box_metrics.csv` (one row per scenario), `single_box_crop_vs_full.csv`, `single_box_roi_vs_full.csv` and a macro-F1 forest plot for each.
2. It fails if a single-box photo is missing from, or appears twice in, any of the 24 files, or matches a row with another case or class.
3. Tests (`tests/test_single_box.py`, CPU, hand-made prediction files): the subset keeps only one-box test photos; every scenario is scored on the same photos and cases; a two-box photo's crops are left out; a missing photo is an error.

---

## 4. Files

| File | Change |
|---|---|
| `src/evaluation/single_box.py` | New |
| `tests/test_single_box.py` | Tests |
| `runbook.md`, `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Command, scope, memo |

---

## 5. Acceptance Check

1. `pytest tests/test_single_box.py` passes (locally and in the Raptor container).
2. The run reports 127 single-box test photos from 57 cases and writes all outputs.
3. Record the results here.

---

## 6. HPC Run (2026-10-21)

- **Tests**: `tests/test_single_box.py` 4 passed locally.
- **Run**: `single_box.py` from commit `409924e` on the evaluated main and `roi_crop` runs; 127 single-box test photos from 57 cases, every photo found once in all 24 files; 2,000 case resamples.

**Per-crop (S25–S32) − full image, test macro-F1 on 127 photos**

| Family | Full image | Per-crop | Difference [95% CI] | Holm p |
|---|---|---|---|---|
| EfficientNet-B0 | 0.767 | 0.841 | +0.074 [−0.052, +0.224] | 1.000 |
| ResNet152 | 0.842 | 0.821 | −0.021 [−0.108, +0.057] | 1.000 |
| MobileNetV3-Small | 0.688 | 0.726 | +0.038 [−0.119, +0.161] | 1.000 |
| GoogLeNet | 0.741 | 0.772 | +0.031 [−0.093, +0.195] | 1.000 |
| VGG16 | 0.706 | 0.721 | +0.015 [−0.125, +0.143] | 1.000 |
| ViT-B/16 | 0.763 | 0.725 | −0.037 [−0.164, +0.097] | 1.000 |
| CNN-SVM | 0.631 | 0.743 | +0.113 [−0.022, +0.251] | 0.832 |
| AlexNet ensemble | 0.689 | 0.770 | +0.081 [−0.050, +0.209] | 1.000 |
| **Pooled (mean of 8)** | 0.728 | 0.765 | **+0.037 [−0.033, +0.107]**, p = 0.307 | – |

- Secondary, pooled: balanced accuracy +0.033 [−0.049, +0.118], p = 0.437; macro AUC +0.030 [−0.005, +0.072], p = 0.098 (CNN-SVM AUC +0.059, Holm p = 0.040). Mean accuracy 81.5% (full) vs 84.3% (per-crop).
- **Union ROI (S02–S16) − full image**, pooled macro-F1 +0.031 [−0.045, +0.099], p = 0.416; 0 of 8 families after Holm (CNN-SVM +0.128, unadjusted p = 0.038, Holm 0.304).

**Reading**
- The crop is higher than the full image in 6 of 8 families and on average (+0.037 macro-F1, +2.8 points accuracy), but no interval excludes zero: on 57 cases the one-input comparison does **not** show a significant gain on its own.
- The size of the pooled gain matches the full test set of Experiment 1 (+0.036) and is below the cross-validation estimates (+0.060 to +0.061), which remain the strongest evidence that cropping helps.
- The two strongest full-image models, ResNet152 and ViT-B/16, do not gain from the crop; the weaker families (CNN-SVM, AlexNet ensemble, EfficientNet-B0) gain most.
- Deck: `05_Presentations/Experiment presentations/Exp 3b - Full image vs one wound crop.pptx` (builder `_build/exp3b_deck.js`).

---

## 7. How to read it

- The subset was chosen after the test results, on the same 62-case test set, so it is exploratory and not independent of earlier choices.
- 57 cases, and only 4 scratch and 7 penetration photos: per-family intervals will be wide. The pooled difference over the 8 families is the main number.
- Single-box photos may be easier or harder than multi-box ones, so these scores describe one-wound photos, not the whole test set.
