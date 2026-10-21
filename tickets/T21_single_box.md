# Ticket T21: Single-Box Test Photos, One Full Image vs One Crop

- **Ticket ID**: `T21` (`T20` is reserved for the Experiment 5 plan)
- **Phase**: Extra analysis, outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T15](T15_summary_and_full_vs_roi.md) (paired case-level bootstrap); [T17](T17_box_roi.md) (the per-box crops); [T18](T18_crop_roi.md) (the per-crop models)
- **Scenarios**: none new; reuses the test predictions of S01–S16 (main run) and S25–S32 (`roi_crop`)
- **Status**: Code done; Raptor run pending

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

## 6. How to read it

- The subset was chosen after the test results, on the same 62-case test set, so it is exploratory and not independent of earlier choices.
- 57 cases, and only 4 scratch and 7 penetration photos: per-family intervals will be wide. The pooled difference over the 8 families is the main number.
- Single-box photos may be easier or harder than multi-box ones, so these scores describe one-wound photos, not the whole test set.
