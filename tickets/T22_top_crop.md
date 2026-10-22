# Ticket T22: One Crop per Photo (the Most Confident) Next to the Full Image

- **Ticket ID**: `T22`
- **Phase**: Extra analysis of the per-crop run (T18), outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T18](T18_crop_roi.md) (the per-crop models S25–S32); [T08](T08_evaluation_module.md) (metrics, case bootstrap)
- **Scenarios**: none new; S25–S32 scored per photo, shown next to S01–S15 (full image)
- **Status**: Code done; Raptor run pending

---

## 1. Goal

Score the per-crop models of Experiment 3 on the same unit as the full-image models: **one input per test photo**. A photo with one box has one crop. A photo with several boxes keeps the crop the model is most confident about, so every one of the 157 test photos has one full-image prediction and one crop prediction. The result is shown side by side with the full image as a description of how well each model performs; there is no paired test.

No model is trained and no new test prediction is made.

---

## 2. Decisions (2026-10-22, after every test result had been seen)

Decided with the researcher for the Experiment 3 presentation, after the test results of Experiments 1–3 and T21 had been seen, so the thesis reports T22 as a **post hoc, descriptive** analysis.

| Item | Decision | Reason |
|---|---|---|
| Photos | All 157 test photos, 62 cases (127 with one crop, 30 with several) | One prediction per photo for both inputs, without dropping multi-wound photos |
| Crop per photo | The crop with the highest top-class probability; ties go to the first crop in file order. Chosen by each model for itself | Uses no label, so the rule could run on new photos; researcher's choice over a random crop |
| Models | Per-crop models S25–S32 (T18), unchanged | Experiment 3 |
| Full image | S01, S03, …, S15 test metrics, unchanged | The reference shown next to the crop |
| Metrics | T08 set per model: accuracy, balanced accuracy, macro precision/recall/F1, macro AUC, per-class precision/recall/F1/AUC, confusion matrices; 95% case-bootstrap intervals (2,000 resamples, seed 42) | Same as every scored scenario |
| Comparison | **None**: no paired bootstrap, no p-values; the two inputs are listed side by side, best and worst named | Researcher decision |

---

## 3. Requirements

1. `photo_top_crop_frame` (`src/training/photo.py`): one row per photo with the most confident crop's probabilities, the crop count and the chosen crop (`top_crop`).
2. `src/evaluation/top_crop.py --crop-root <exp>/roi_crop --main-root <exp>/main` writes
   - `roi_crop/crop/<family>/top_crop/` (photo-level prediction files and the T08 metric files; per-image files stay on Raptor);
   - `roi_crop/summary/top_crop_table.csv` (full image and top crop per family, with intervals) and `top_crop_per_class.csv` (per-class precision, recall, F1, AUC for both inputs).
3. It fails if a crop is missing from the per-box manifest, or if the top-crop test photos differ from the full-image ones (cases and photos per class).
4. Tests (`tests/test_top_crop.py`, CPU, hand-made files): the most confident crop is kept even when it is wrong; the table lists both inputs on the same photos; a crop missing from the manifest is an error.

---

## 4. Files

| File | Change |
|---|---|
| `src/training/photo.py` | `photo_top_crop_frame` |
| `src/evaluation/top_crop.py` | New |
| `tests/test_top_crop.py` | Tests |
| `runbook.md`, `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Command, scope, memo |

---

## 5. Acceptance Check

1. `pytest tests/test_top_crop.py` passes (locally and in the Raptor container).
2. Every family is scored on 157 test photos from 62 cases for both inputs.
3. Record the results here.

---

## 6. How to read it

- Choosing the crop by the model's own confidence favours crops the model finds easy; it is a deployable rule, not an oracle, because no label is used.
- The two inputs are scored on the same photos but are not tested against each other; differences between them are descriptive.
- Same 62-case test set as every other analysis; post hoc.
