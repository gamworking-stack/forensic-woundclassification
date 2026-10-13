# Ticket T12b: 5-Fold Grouped Cross-Validation on Development Data

- **Ticket ID**: `T12b`
- **Phase**: Phase 5 (Experiments, optional item)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §5 Phase 5 (optional 5-fold grouped CV on train+validation); [T08b](T08b_lr_tuning.md); [T15](T15_summary_and_full_vs_roi.md)
- **Scenarios**: all (S01–S16)
- **Status**: Completed

---

## 1. Goal

Estimate each scenario's performance as a mean ± SD over 5 case-grouped folds of the **development data (train + validation)**, and repeat the paired full vs ROI comparison on the out-of-fold predictions of all development cases. The test split is never loaded.

---

## 2. Decision (2026-10-14)

This analysis was decided **after** the test results (T08 §7, T15 §7) were seen. It is **supplementary**: it changes no model, setting or test result, and the thesis reports it as a post hoc robustness analysis.

| Item | Decision | Reason |
|---|---|---|
| Data | Train + validation rows of the frozen split; test rows dropped | ROADMAP: CV on development data only |
| Folds | 5 folds by `pseudonimized_id`, built once by a seeded search (seed 42, 50,000 trials) for class balance, every class in every fold; frozen in `data/splits/cv_folds.csv` | Same approach as T02; one fold assignment shared by every scenario and both paths |
| Fold balance | Scored on the ROI manifest's development rows (the matched data both paths train on, `full_scope = matched`) | Both paths train on the same images |
| Per fold | Train on 4 folds, predict the held-out fold | |
| Settings | Each scenario's main-run settings, **not re-tuned**: S01–S12 the T08b chosen learning rate; S13–S14 the T06 chosen SVM; S15–S16 seeds 43–45 at lr 3e-5 | No tuning against the CV folds |
| Epochs | **Fixed** at the main run's best epoch (S01–S12: selected grid run; S15–S16: each member's best epoch); the model after the last epoch predicts the held-out fold | Early stopping on the held-out fold would select on the data being scored |
| Metrics | Per fold: macro-F1 (primary), balanced accuracy, macro AUC, accuracy; mean ± SD over folds. Pooled out-of-fold metrics over all development cases | |
| Paired comparison | T15 method on the out-of-fold predictions: 2,000 case resamples (seed 42) shared by every scenario, ROI − full per family, Holm over 8 families, pooled mean | Uses 356 cases instead of 62 |
| Caveat | The learning rate, SVM setting and epoch count were chosen on the main validation split, which lies inside these folds, so CV estimates are slightly optimistic; the paired full vs ROI differences are less affected because both paths share it | Report in the thesis |

---

## 3. Requirements & Business Logic

1. **Folds** (`src/data/create_cv_folds.py`): read both manifests, drop test rows, assign every development case to one of 5 folds (case counts as equal as possible), with every class present in every fold (images and cases) on the ROI rows. Write `data/splits/cv_folds.csv` (`pseudonimized_id, fold`) and `data/splits/cv_folds_report.json` (per fold and class: images, cases). Assert no case is in two folds and no test case is included.
2. **CV mode in the training code** (`--cv-folds <csv> --fold <k>`, `train.py` and `cnn_svm.py`): test rows are dropped, rows of fold k become `validation`, the other folds `train`; the test split is never loaded or predicted. The class-coverage check covers train and validation only.
3. **Fixed epochs** (`train.py --checkpoint last`): predictions use the weights after the last epoch instead of `best.pt`; with `--patience` equal to `--max-epochs` training runs the full count. Resume works as in T05.
4. **CNN-SVM** (`cnn_svm.py --svm-params '<json>'`): skip the grid and fit the given setting on the 4 training folds.
5. **Ensemble** (`ensemble.py --splits validation`): soft-vote only the held-out-fold predictions.
6. **Plan and submission**:
   - `src/training/cv_plan.py` reads the main run (`tuning_summary.csv`, `ensemble_members.csv`, CNN-SVM `run_config.json`) and writes one line per job (scenario, family, path, fold, lr, epochs, seed, SVM setting, output folder).
   - `scripts/hpc/submit_cv.sh` submits every job of the plan that is neither finished (`validation_predictions.csv`) nor queued, through `train_single.sbatch`, plus an `afterok` combine job per ensemble fold.
   - Outputs: `<project>/experiments/cv/<full|roi>/<family>/fold_<k>/` (ensemble: `.../alexnet_ensemble/fold_<k>/member_0<j>/`).
7. **Summary** (`src/evaluation/cv_summary.py --cv-root <project>/experiments/cv`): write `summary/cv_table.csv` (per scenario: mean, SD and per-fold values; pooled out-of-fold metrics), `summary/cv_full_vs_roi.csv` and `summary/cv_full_vs_roi_macro_f1.png`. Every scenario must have all 5 folds; all scenarios must cover the same development cases.
8. **Automated Testing** (CPU, synthetic data):
   - Folds: deterministic for a seed, every class in every fold, no case in two folds, no test case.
   - CV mode: fold k becomes validation, no test file is written, `--checkpoint last` predicts with the last epoch.
   - CNN-SVM with `--svm-params` writes only validation predictions and no tuning table.
   - Ensemble `--splits validation` works without test files.
   - Plan: chosen lr, best epochs and SVM setting are read from a synthetic main run.
   - Summary: identical full and ROI give 0 difference; a missing fold raises.

---

## 4. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/data/create_cv_folds.py` | Fold assignment |
| `data/splits/cv_folds.csv`, `cv_folds_report.json` | Frozen folds |
| `src/training/data.py`, `config.py`, `train.py` | CV mode, `--checkpoint last` |
| `src/training/cnn_svm.py` | CV mode, `--svm-params` |
| `src/training/ensemble.py` | `--splits` |
| `src/training/cv_plan.py` | Job plan from the main run |
| `scripts/hpc/submit_cv.sh` | Submission |
| `scripts/slurm/ensemble_combine.sbatch` | Pass `--splits` |
| `src/evaluation/cv_summary.py` | CV table and paired comparison |
| `tests/test_cv.py` | Tests above |
| `runbook.md` | Commands |

---

## 5. Acceptance Check

1. `pytest tests/` passes.
2. On Raptor, every job of the plan finishes; each of the 16 scenarios has 5 held-out-fold prediction files covering all development cases once.
3. `cv_summary.py` writes the three files; the figure is inspected.
4. Record the CV results here and note in T12 that the optional CV was run.

---

## 7. HPC Run (2026-10-14)

`submit_cv.sh`: 110 jobs (60 single-model, 10 CNN-SVM, 30 AlexNet members, 10 combine), 03:34–05:54, all completed; every run recorded commit `760e799` with `git_dirty: false`. `pytest tests/test_cv.py` passed in the Raptor container (8 passed). `cv_summary.py`: 16 scenarios x 5 folds, 356 development cases (943 images) out of fold, 2,000 paired resamples. The figure was inspected (same layout and scale as the T15 test figure).

**Per scenario: macro-F1 over 5 folds (mean ± SD) and out of fold**

| Family | Full | ROI |
|---|---|---|
| EfficientNet-B0 | 0.730 ± 0.035 (0.741) | 0.774 ± 0.035 (0.788) |
| ResNet152 | 0.761 ± 0.033 (0.769) | 0.789 ± 0.029 (0.801) |
| MobileNetV3-Small | 0.636 ± 0.026 (0.650) | 0.729 ± 0.027 (0.744) |
| GoogLeNet | 0.736 ± 0.025 (0.747) | 0.737 ± 0.041 (0.749) |
| VGG16 | 0.753 ± 0.038 (0.768) | 0.761 ± 0.027 (0.773) |
| ViT-B/16 | 0.706 ± 0.062 (0.716) | 0.774 ± 0.031 (0.787) |
| CNN-SVM | 0.529 ± 0.020 (0.537) | 0.717 ± 0.052 (0.732) |
| AlexNet ensemble | 0.643 ± 0.050 (0.662) | 0.695 ± 0.032 (0.708) |

**Paired full vs ROI, out-of-fold macro-F1 (primary)**

| Family | Full | ROI | ROI − full [95% paired CI] | p | Holm p |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0.741 | 0.788 | +0.047 [+0.009, +0.079] | 0.017 | 0.085 |
| ResNet152 | 0.769 | 0.801 | +0.032 [−0.009, +0.073] | 0.145 | 0.435 |
| MobileNetV3-Small | 0.650 | 0.744 | +0.094 [+0.052, +0.133] | <0.001 | <0.001 |
| GoogLeNet | 0.747 | 0.749 | +0.002 [−0.039, +0.041] | 0.935 | 1.000 |
| VGG16 | 0.768 | 0.773 | +0.005 [−0.032, +0.042] | 0.810 | 1.000 |
| ViT-B/16 | 0.716 | 0.787 | +0.072 [+0.029, +0.114] | <0.001 | <0.001 |
| CNN-SVM | 0.537 | 0.732 | +0.195 [+0.143, +0.242] | <0.001 | <0.001 |
| AlexNet ensemble | 0.662 | 0.708 | +0.046 [−0.001, +0.090] | 0.055 | 0.220 |
| **All families (mean)** | 0.699 | 0.760 | **+0.061 [+0.036, +0.085]** | <0.001 | – |

p < 0.001 means no bootstrap resample out of 2,000 crossed 0. **Secondary (pooled)**: balanced accuracy +0.059 [+0.033, +0.083], p < 0.001; macro AUC +0.016 [+0.005, +0.026], p = 0.002. Holm p < 0.05 on balanced accuracy for EfficientNet-B0, MobileNetV3-Small, ViT-B/16 and CNN-SVM, and on macro AUC for CNN-SVM.

**Reading**
- On 356 development cases the ROI path is better on average (+0.061 macro-F1), with three families (MobileNetV3-Small, ViT-B/16, CNN-SVM) significant after Holm; no family favours the full image. GoogLeNet and VGG16 show no difference.
- This agrees in direction with the test set (T15: +0.036 [−0.031, +0.091]), whose 62 cases could not resolve it. ResNet152 is −0.058 on test but +0.032 out of fold, so its test result is compatible with noise.
- Limits: post hoc and supplementary (§2); the learning rate, SVM setting and epoch counts were chosen on the main validation split, which lies inside these folds, so absolute CV scores are optimistic (both paths share this); the bootstrap resamples cases, not training runs, so run-to-run training variation is not in the intervals; internal data from one hospital.
