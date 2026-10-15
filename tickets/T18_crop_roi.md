# Ticket T18: Per-Crop ROI Experiment (each crop its own sample)

- **Ticket ID**: `T18`
- **Phase**: Extra experiment, outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T17](T17_box_roi.md) (the same crops); [T08](T08_evaluation_module.md) (metrics, case bootstrap); [T08b](T08b_lr_tuning.md) (tuning); [T16](T16_final_verification.md) (verification)
- **Scenarios**: S25–S32 (input path `crop`, the 8 families in S01–S16 order)
- **Status**: Completed (2026-10-15)

---

## 1. Goal

Classify every wound crop on its own and report the result per crop. T18 uses the T17 crops, but each crop is a sample in training, model selection and evaluation. Nothing is averaged per photo, and the result is not compared with any other run. The experiment changes nothing in the main, T17 or CV runs: it has its own experiments root.

---

## 2. Decisions (2026-10-15, before any per-crop model was trained)

Decided with the researcher after the test results of the main study and T17 had been seen, so the thesis reports T18 as a post hoc additional analysis.

| Item | Decision | Reason |
|---|---|---|
| Crops | The T17 crops unchanged: `data/roi_box/roi_manifest.csv`, `derived_data/roi_box_p008` (8% padding), 1,667 crops: train 1,177 / validation 277 / test 213, from 1,100 photos and 418 cases | Same images as T17; only the unit of analysis changes |
| Label and split | A crop takes its photo's label and inherits its photo's case-level split; leakage check on cases | As T17 |
| Training weight | Every crop counts once: **no** 1/n photo weight. Class weights = inverse class frequency of the **training crops**, normalised to mean 1 (T05 rule) | Researcher decision: pure crop-level training |
| Model selection | Early stopping and learning-rate choice on **crop-level** validation macro-F1 (277 crops); ties go to the lower validation loss | Selection on the unit that is evaluated |
| Families and budget | All 8, the same rules as S01–S24: lr grid {1e-5, 3e-5, 1e-4} for S25–S30, CNN-SVM 29-point grid (unweighted SVM, balanced class weights on crop counts), AlexNet ensemble seeds 43–45 at lr 3e-5; max 200 epochs, patience 25, batch 32, seed 42 | Researcher decision: all families |
| Primary outcome | Test **crop-level** macro-F1 (213 crops, 62 cases) with a 95% case-level bootstrap interval: 2,000 resamples of cases, all crops of a case kept together, seed 42 | Researcher decision; crops of one patient are not independent |
| Secondary | Accuracy, balanced accuracy, macro precision/recall, macro AUC; per-class precision, recall, F1 and AUC; confusion matrices | T08 metric set |
| Comparisons | **None**: no photo level, no paired test against the full image, union ROI or per-box runs | Researcher decision: per-crop results only |
| Supplementary | Verification only (`src/evaluation/verify_crop.py`). No cross-validation, no new Grad-CAM | Researcher decision |
| Reporting | Results files and per-class Excel; this ticket, `docs/methodology_two_experiments.md` and README; thesis deck (2 slides in Results); AFSN deck (backup only); AFSN abstract unchanged | Researcher decision |
| Caveat | Each crop counts once, so a photo with several wounds weighs more than a one-wound photo: 30 of the 157 test photos give more than one crop (up to 35 in one photo). Stated with every result | Fact of the design |

---

## 3. Requirements

1. **Training** (`train.py` / `cnn_svm.py --input-path crop`): images from `roi_relative_path`; no loss weights; validation macro-F1 and every prediction file per crop. Prediction files have one row per crop, with `relative_path` the crop and **no** `source_relative_path`, so every evaluation stays crop level. `run_config.json` records `selection_unit: crop`, `loss_weights: none`.
2. **Evaluation**: `evaluate.py --experiments-root <exp>/roi_crop` evaluates `crop/<family>/` as S25–S32 with the T08 metrics and case bootstrap.
3. **Summary** (`src/evaluation/crop_summary.py --crop-root <exp>/roi_crop`): `summary/crop_summary_table.csv`, one row per scenario with test crops, cases, the T15 table metrics and their intervals.
4. **Verification** (`src/evaluation/verify_crop.py`):
   - files exist;
   - the files are crop level and hold only the frozen split's cases;
   - every scenario is tested on the manifest's 213 test crops;
   - each selection is reproduced from validation data, and the networks select per crop without weights;
   - the budget equals the main run's;
   - every number and table is recomputed.
5. **HPC**: `train_single.sbatch` accepts `INPUT_PATH=crop` (the per-box manifest and crops); `submit_all.sh` names `crop` jobs S25–S32 (`PATHS=crop ... roi_crop`).
6. **Tests** (`tests/test_crop_roi.py`, CPU, synthetic):
   - training on `crop` writes crop rows without a photo column, with crop-count class weights and crop-level selection;
   - CNN-SVM on `crop` works;
   - evaluate names the scenarios S25–S32;
   - `crop_summary.py` and `verify_crop.py` pass end to end;
   - verification catches an edited number and photo-level files.

---

## 4. Files

| File | Change |
|---|---|
| `src/training/config.py`, `data.py`, `train.py`, `cnn_svm.py` | `crop` input path |
| `src/evaluation/evaluate.py` | `crop` scenarios S25–S32 |
| `src/evaluation/crop_summary.py`, `src/evaluation/verify_crop.py` | New |
| `scripts/slurm/train_single.sbatch`, `scripts/hpc/submit_all.sh`, `scripts/hpc/submit_alexnet_ensemble.sh` | `crop` path |
| `tests/test_crop_roi.py` | Tests |
| `runbook.md`, `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Commands, scope, memo |

---

## 5. Acceptance Check

1. `pytest tests/` passes (locally and in the Raptor container).
2. 8 of 8 scenarios finish; `evaluate.py` evaluates S25–S32 on 213 test crops from 62 cases; `crop_summary.py` writes the table.
3. `verify_crop.py` reports no FAIL.
4. Record the results here.

---

## 6. HPC Run (2026-10-15)

- **Tests**: `pytest tests/` 129 passed locally; `tests/test_crop_roi.py` 6 passed in the Raptor container.
- **Training** (jobs 230013–230041, commit `8da4750`, `git_dirty: false`): 29 of 29 completed on the first attempt.
- **Evaluation** (job 230055): `evaluate.py`, `crop_summary.py` and `verify_crop.py` in one job. 8 scenarios, 213 test crops from 62 cases, 2,000 case resamples.
- **Verification**: 157 PASS, 0 FAIL, 1 NOTE: all 21 trained networks ran from commit `8da4750`.

**Choices on validation (crop-level macro-F1, 277 crops)**

| Scenario | Family | Chosen | Best epoch | Validation macro-F1 |
|---|---|---|---|---|
| S25 | EfficientNet-B0 | lr 1e-4 | 8 | 0.749 |
| S26 | ResNet152 | lr 3e-5 | 10 | 0.781 |
| S27 | MobileNetV3-Small | lr 1e-4 | 12 | 0.770 |
| S28 | GoogLeNet | lr 3e-5 | 61 | 0.736 |
| S29 | VGG16 | lr 3e-5 | 19 | 0.725 |
| S30 | ViT-B/16 | lr 1e-4 | 11 | 0.763 |
| S31 | CNN-SVM | RBF, C 100, gamma 1e-6 | – | 0.698 |
| S32 | AlexNet ensemble | seeds 43–45 (lr 3e-5) | 26 / 31 / 14 | 0.706 / 0.705 / 0.679 |

Grid edges: S25, S27 and S30 chose the upper limit (1e-4); S31 sits on the smallest gamma, as in S13 and S23. The grids were not widened.

**Test, per crop (213 crops, 62 cases)**

| Scenario | Family | Accuracy | Balanced acc. | Macro-F1 [95% CI] | Macro AUC |
|---|---|---|---|---|---|
| S25 | EfficientNet-B0 | 0.873 | 0.841 | **0.797** [0.707, 0.854] | 0.962 |
| S26 | ResNet152 | 0.864 | 0.804 | 0.767 [0.677, 0.839] | 0.952 |
| S27 | MobileNetV3-Small | 0.826 | 0.739 | 0.730 [0.608, 0.786] | 0.965 |
| S28 | GoogLeNet | 0.845 | 0.769 | 0.719 [0.638, 0.778] | 0.947 |
| S29 | VGG16 | 0.859 | 0.745 | 0.727 [0.626, 0.799] | 0.963 |
| S30 | ViT-B/16 | 0.845 | 0.721 | 0.715 [0.612, 0.782] | 0.954 |
| S31 | CNN-SVM | 0.812 | 0.734 | 0.694 [0.587, 0.787] | 0.972 |
| S32 | AlexNet ensemble | 0.840 | 0.776 | 0.737 [0.635, 0.809] | 0.947 |

**Per class, mean F1 over the 8 families** (test crops / cases): gunshot wound 0.928 (73 / 17), graze wound 0.926 (51 / 20), contusion 0.923 (34 / 14), laceration 0.735 (22 / 13), scratch 0.599 (6 / 5), cut wound 0.547 (20 / 7), penetration wound 0.491 (7 / 5).

**Reading**
- EfficientNet-B0 is the best per-crop model (macro-F1 0.797, accuracy 87.3%), followed by ResNet152 (0.767). The other six are 0.69–0.74, with overlapping intervals.
- Macro AUC is 0.947–0.972 for every model, so the crop probabilities rank the classes well even where the top-1 class is wrong.
- Gunshot, graze and contusion crops are classified well. Cut wound, scratch and penetration are hardest; scratch and penetration have only 6 and 7 test crops from 5 cases each, so their per-class F1 moves by about 0.1–0.2 with one crop.
- Limits:
  - A photo with several wounds counts once per crop (30 of 157 test photos give more than one crop; gunshot has 73 crops from 17 cases).
  - It is the same 62-case test set as the other experiments.
  - The analysis is post hoc relative to the proposal, with one training seed per setting.
  - Crops still need manual boxes.

