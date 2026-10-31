# Ticket T29: Inconclusive Below a Confidence Threshold ("Halloween experiment")

- **Ticket ID**: `T29`
- **Phase**: Extra analysis of the T23 test results (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T08](T08_evaluation_module.md) (metrics, bootstrap), [T25](T25_patient_cases.md) (patient cases), [T28](T28_precision_recall.md) (macro precision and recall)
- **Scenarios**: none new; the T23 test predictions of S01–S15 (full image) and S25–S32 (per-crop) re-scored
- **Status**: Done (laptop, 2026-10-31; results §4)

---

## 1. Goal

The researcher asked (2026-10-31) for the test results again with the same metrics, but with a confidence threshold: if the model's confidence in its class does not reach about 60%, the image is **inconclusive**. Show how many images are lost to inconclusive and the accuracy against the true class, in a new folder "Halloween experiment".

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Confidence | Top softmax probability of the saved prediction (CNN-SVM: its probability output; AlexNet ensemble: the soft-vote mean) | What every prediction file holds |
| Threshold | 0.60, fixed by the researcher; top probability < 0.60 = inconclusive. Point estimates also at 0.5, 0.7, 0.8, 0.9 and a 0–0.95 curve | Not tuned on the test set |
| Scenarios | Full image S01–S15 (photos) and per-crop S25–S32 (crops), as in journal Tables 2 and 4 | Same tables as before |
| Metrics | On the conclusive images: accuracy, balanced accuracy, macro precision, macro recall, macro-F1, macro AUC (T08/T28 definitions; averages over the classes present). Also coverage, inconclusive images and how many of them were right or wrong, and accuracy over every image with inconclusive counted as not correct | Same metrics, plus what the threshold costs |
| Intervals | 95% patient-level bootstrap (T25 copies), the T08 resamples (2,000, seed 42); each resample is scored on its conclusive rows. Pooled = mean of the 8 families, interval from per-resample means (T15) | As Tables 2 and 4 |
| Check | At threshold 0 the script must reproduce every saved `test_metrics.json` value and interval, or it stops | Same numbers as before when nothing is inconclusive |
| Privacy | Prediction copies in a temporary folder, deleted after the run; outputs hold no per-image rows and the script refuses an output folder inside the repo | T23 §3 |

## 3. Files

| File | Change |
|---|---|
| `src/evaluation/confidence_threshold.py` | New |
| `tests/test_confidence_threshold.py` | Counts and accuracy on a hand-made example, threshold 0 equals `compute_metrics` and `bootstrap_draws`, output inside the repo refused |
| `runbook.md` (T23 step 12), `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Command, scope, memo |

Outputs in `../results_figures/Halloween experiment/`: `threshold_summary.csv` (per scenario and pooled, with intervals), `threshold_per_class.csv` (per true class), `threshold_confusion.csv` (confusion counts with an `inconclusive` column), `threshold_sweep.csv` (0, 0.5–0.9, point estimates), `threshold_coverage.png` (pooled accuracy against coverage).

## 4. Run log and results

- 2026-10-31 (laptop): the 16 test prediction files copied from Raptor, rewritten per patient (`patient_rescore.py`, 391 patient cases), scored (about 5 minutes), deleted. Threshold 0 reproduced every saved value and interval (full and per-crop).
- `pytest tests/test_confidence_threshold.py tests/test_precision_recall.py`: 4 passed (Python 3.11, CPU).

Threshold 0.60, test set 149 photos / 276 crops, 55 patients. Pooled = mean of 8 families [95% patient bootstrap]:

| Input | Inconclusive (of which right / wrong) | Accuracy, no threshold | Accuracy, conclusive | Macro-F1, conclusive | Accuracy, all (inconclusive = not correct) |
|---|---|---|---|---|---|
| Full image | 35.1 of 149 (23.6%) (18.8 / 16.4) | 0.791 | 0.866 [0.813, 0.913] | 0.748 [0.683, 0.816] | 0.665 [0.602, 0.730] |
| Per-crop | 46.8 of 276 (16.9%) (24.0 / 22.8) | 0.806 | 0.866 [0.776, 0.933] | 0.744 [0.648, 0.854] | 0.719 [0.634, 0.792] |

- Accuracy on the images the models answer rises by about 0.06–0.08, but about half of the images set aside were classified correctly, so roughly one correct answer is lost for each error avoided.
- The share set aside depends on the family's calibration: full image 9% (EfficientNet-B0, ViT-B/16) to 45% (CNN-SVM); per-crop 9% to 30% (AlexNet ensemble). A single 0.60 cut-off means different things for different families.
- ResNet152 (full) sets aside 13 photos, 12 of them correct: its accuracy does not rise (0.893 → 0.890).
- Per class (full image, pooled), penetration wounds (0.36) and scratches (0.51) stay least accurate after thresholding, and cut wounds on per-crop (0.49).
- Macro averages run over the classes still present among the conclusive images. Macro AUC is computed on a subset chosen by the model's own confidence, so it is not comparable with Table 2's AUC.
- Post hoc and descriptive: the threshold was not chosen on separate data, and the comparison with no threshold is not a paired test.
