# Ticket T25: One Case per Patient for the T23 Intervals

- **Ticket ID**: `T25`
- **Phase**: Re-scoring of the T23 rerun, outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T23](T23_relabel_rerun.md) §3 (grouping), [T08](T08_evaluation_module.md) (case bootstrap), [T24](T24_subgroups.md) (subgroups)
- **Scenarios**: none new; the T23 predictions of S01–S16, S25–S32 and the per-box photo files re-scored
- **Status**: Done (laptop, 2026-10-28; results §5)

---

## 1. Goal

T23 joins photos that share a sheet case number or a `patient_id` into one group, and every split, fold and bootstrap uses that group. Because the sheet's case numbers are misaligned from case 316 on, 21 groups join several case numbers or patients (17 hold more than one patient, up to 10 patients and 36 photos). The researcher asked (2026-10-28) to count cases as patients instead: one case per `patient_id`.

---

## 2. Decisions (researcher, 2026-10-28)

| Item | Decision | Reason |
|---|---|---|
| Case | One case per `patient_id`; a photo with no patient ID (`n/a`, blank; 8 photos) counts its sheet case number as the patient | Option B: patients as the unit, no sheet edit |
| Split, folds, models | Unchanged. No patient crosses a T23 group, so every patient is already in one split and one CV fold | No retraining |
| What is redone | The test case-bootstrap intervals (T08), the paired full vs ROI comparison (T15), the per-crop table (T18) and the T24 subgroups, on copies of the prediction files with the case column replaced | Point estimates cannot change; only the resampling unit does |
| Pseudonym | `patient_case` numbered from 1 by each patient's smallest group, sheet case and photo; `patient_id` is never written | As `pseudonimized_id` |
| Sheet | `../thesis_data_27_oct_corrected.csv` (T24 §7) | The newest sheet with `patient_id` |

---

## 3. Files

| File | Change |
|---|---|
| `src/data/patient_cases.py` | New: sheet + T23 split manifest → `patient_cases.csv`; refuses a patient that crosses groups |
| `data/relabel_2026_10_23/splits/patient_cases.csv` | New: `patient_case` of all 1,041 photos (393 patients); 1,032 photos (391 patients) after T26 |
| `src/evaluation/patient_rescore.py` | New: copies `*_predictions.csv` with `pseudonimized_id` set to `patient_case`, plus a matching split manifest; output outside the repo |
| `tests/test_patient_cases.py` | Numbering, a patient across groups refused, full/ROI/crop rows rewritten, a row of another group refused, output inside the repo refused |
| `runbook.md` | Step 10 |

---

## 4. Run log

- 2026-10-28 (laptop): `patient_cases.py` → 393 patients (train 277, validation 61, test 55) from 336 groups; 0 patients in more than one split or fold. The 56 prediction files (main full/roi and roi_crop test + validation, roi_box photo test) copied from Raptor to a temporary folder, rewritten, re-evaluated (`evaluate.py`, `summarize.py`, `crop_summary.py`, `subgroup.py`), and deleted. Outputs without per-image rows: `../results_figures/relabel_2026_10_23_patient/{main,roi_crop,subgroup}/`. The group-level results in `../results_figures/relabel_2026_10_23/` are kept.
- `pytest tests/test_patient_cases.py tests/test_subgroup.py`: 10 passed (Python 3.11, CPU).

---

## 5. Results (test: 149 photos, 55 patients; 276 crops)

- Every point estimate is identical to T23 (checked on `summary_table.csv` and `crop_summary_table.csv`). Intervals move by a few thousandths; mean width 0.157 → 0.159 (full/union) and 0.170 → 0.167 (per-crop).
- Pooled union ROI − full macro-F1: +0.067 [−0.012, +0.126] (T23 groups: [−0.006, +0.123]).
- Subgroups (T24): with patients as the unit, the living group is 6 patients, not 2 groups. Living − autopsy full-image accuracy −0.196 [−0.382, +0.093] (groups: [−0.308, −0.079]); no pooled subgroup difference on full image or per-crop excludes 0. The T24 §7 statement "living = 2 patients" counted groups and is corrected here.
- The journal tables (Claude Doc) report these intervals and patient counts.
