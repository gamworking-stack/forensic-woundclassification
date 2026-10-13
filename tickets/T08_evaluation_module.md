# Ticket T08: Evaluation Module

- **Ticket ID**: `T08`
- **Phase**: Phase 3 (Training Code)
- **Spec Reference**: [specs/phase3_training_code.md](../specs/phase3_training_code.md) §5
- **Scenarios**: all (S01–S16)
- **Status**: Completed

---

## 1. Goal

Implement `src/evaluation/` to compute every metric in the proposal from the per-image prediction files that T05, T06 and T07 write. Metrics are computed **only from `validation_predictions.csv` and `test_predictions.csv`, never from a live model**, so all 16 scenarios are scored by one implementation. Each scenario's test metrics are reported with case-level confidence intervals and case counts, because the test set has only 62 cases.

---

## 2. Requirements & Business Logic

1. **Input checks** (per prediction file):
   - Columns as in spec §4.4. One row per image (no duplicate `relative_path`). `split` matches the file name.
   - `true_class` values are within the class list. The class list comes from the `prob_*` columns, in file order.
   - Probabilities are non-negative and sum to 1 (±1e-4) per row.
   - `predicted_class` equals the argmax of the probabilities.
2. **Metrics** (`metrics.py`), for each split:
   - Overall: accuracy, balanced accuracy, macro and weighted precision, recall (sensitivity) and F1, macro and weighted one-vs-rest ROC AUC, log loss (unweighted, from the stored probabilities, so it is comparable across families including CNN-SVM).
   - Class-wise: precision, recall, F1, one-vs-rest AUC, support in images **and cases**.
   - Confusion matrix: counts, rows = true class, columns = predicted, fixed class order.
   - Counts: images and cases in the split.
3. **Case-level bootstrap confidence intervals** (test split):
   - Resample **cases** (`pseudonimized_id`) with replacement, keeping all images of a drawn case together; 2,000 resamples, seed 42; percentile 95% CI.
   - For accuracy, balanced accuracy, macro F1 and macro AUC.
   - Macro averages in a resample run over the classes present in it, so a resample that misses a rare class (scratch and penetration wound have 5 test cases each, about 1% of resamples) leaves that class out instead of scoring it 0. This is how balanced accuracy is defined anyway. Report the number of resamples used per metric.
4. **Outputs per scenario folder** (`evaluate.py --run-dir`), for `<split>` = validation and test:
   - `<split>_metrics.json`: overall metrics, counts, and (test only) CIs.
   - `<split>_classification_report.csv`: class-wise table plus macro and weighted rows.
   - `<split>_confusion_matrix.csv` and `<split>_confusion_matrix.png`. The PNG shows counts with row percentages, using matplotlib's Agg backend.
5. **All scenarios at once** (`evaluate.py --experiments-root <run>`):
   - Find scenario folders as `<run>/<full|roi>/<family>/` that contain prediction files, and evaluate each.
   - Write `<run>/evaluation_index.csv`: scenario ID (spec §5.1), input path, family, test images, test cases, test macro F1 (with CI), and status (`evaluated`, or `missing` for any of the 16 expected scenarios without prediction files).
   - **Fairness check**: every evaluated scenario's test set must have the same cases and the same images per class. A mismatch is an error naming the scenarios. Runs with other scopes, such as the all-1,106 full sensitivity run, go in their own `<run>` folder.
   - Ensemble member folders (`alexnet_ensemble/member_0k`) are not scenarios and are skipped.
6. **Automated Testing** (`tests/test_evaluation.py`, CPU):
   - Perfect predictions give accuracy, macro F1 and AUC of 1.0 and a diagonal confusion matrix.
   - A hand-built 3-class example matches hand-computed precision, recall, F1 and confusion matrix.
   - Bootstrap: same seed gives the same CI; images of one case always move together; a resample without a class leaves it out of the macro averages instead of scoring it 0.
   - Input checks: probabilities not summing to 1, duplicate rows, a mismatched argmax and unknown classes each raise.
   - Fairness check raises when two scenarios have different test cases.
   - `--experiments-root` on a synthetic run tree lists evaluated and missing scenarios and skips member folders.
   - The confusion matrix PNG is written.

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/evaluation/__init__.py` | Package marker |
| `src/evaluation/metrics.py` | Input checks, metrics, class-wise table, confusion matrix, case bootstrap |
| `src/evaluation/plots.py` | Confusion matrix figure |
| `src/evaluation/evaluate.py` | CLI: one scenario, or every scenario in a run with the index and fairness check |
| `tests/test_evaluation.py` | Tests above |
| `runbook.md` | Evaluate command |

ROC curve figures are T13 and the cross-scenario summary table is T15. Both read the files written here and the prediction CSVs.

---

## 4. CLI Interface

```bash
# One scenario
python src/evaluation/evaluate.py \
  --run-dir /project/a260012_FORWCL/experiments/main/roi/cnn_svm

# Every scenario in a run, plus evaluation_index.csv and the fairness check
python src/evaluation/evaluate.py \
  --experiments-root /project/a260012_FORWCL/experiments/main
```

On Raptor, a CPU job is enough:
```bash
srun -p compute -c 2 --mem=8G -t 0:30:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/evaluate.py --experiments-root /project/a260012_FORWCL/experiments/main
```

---

## 5. Acceptance Check

1. `pytest tests/` passes.
2. On Raptor, `--experiments-root .../experiments/main` evaluates every scenario finished so far, lists the rest as `missing`, and passes the fairness check (157 test images, 62 cases on both paths).
3. Sanity check against the earlier runs: the CNN-SVM full-image test metrics computed here are in the range of EXP003. This is a range check only: EXP003 used all 1,106 images and a different grid.
4. Tick T08 in `ROADMAP.md` §9.

---

## 6. First HPC Run

- **Date**: 2026-10-01, `--experiments-root /project/a260012_FORWCL/experiments/main`: 4 scenarios evaluated, 12 missing, fairness check passed.
- Test macro-F1 [95% case-bootstrap CI]: S13 CNN-SVM full 0.585 [0.479, 0.667], S14 CNN-SVM ROI 0.725 [0.605, 0.823], S15 AlexNet ensemble full 0.662 [0.553, 0.748], S16 AlexNet ensemble ROI 0.728 [0.607, 0.817].
- Range check passed: S13 0.585 vs EXP003 0.580 (EXP003 used all 1,106 images and a smaller grid).
- From here on no setting of these scenarios is changed, because their test results have been seen.

## 7. Full HPC Run (all 16 scenarios)

- **Date**: 2026-10-14, after T08b: 16 scenarios evaluated, 0 missing, each on the same 157 test images from 62 cases, fairness check passed (every scenario tested on the same cases and images per class).
- Test macro-F1 [95% case-bootstrap CI]:

| Family | Full | ROI |
|---|---|---|
| EfficientNet-B0 (S01/S02) | 0.733 [0.613, 0.814] | 0.742 [0.631, 0.809] |
| ResNet152 (S03/S04) | 0.783 [0.681, 0.861] | 0.726 [0.616, 0.812] |
| MobileNetV3-Small (S05/S06) | 0.653 [0.557, 0.737] | 0.727 [0.613, 0.800] |
| GoogLeNet (S07/S08) | 0.705 [0.579, 0.783] | 0.736 [0.631, 0.796] |
| VGG16 (S09/S10) | 0.699 [0.600, 0.779] | 0.715 [0.612, 0.785] |
| ViT-B/16 (S11/S12) | 0.754 [0.647, 0.827] | 0.761 [0.671, 0.824] |
| CNN-SVM (S13/S14) | 0.585 [0.479, 0.667] | 0.725 [0.605, 0.823] |
| AlexNet ensemble (S15/S16) | 0.662 [0.553, 0.748] | 0.728 [0.607, 0.817] |

- S13–S16 match the first run exactly, as expected (same prediction files, same bootstrap seed).
- ROI is higher in 7 of 8 families (ResNet152 is lower), but every pair of intervals overlaps; the paired full-vs-ROI comparison on the same test cases belongs to T15.
