# Ticket T26: Drop the Photos Without a CVAT Box from the T23 Dataset

- **Ticket ID**: `T26`
- **Phase**: Data clean-up of the T23 dataset version (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T23](T23_relabel_rerun.md) §4 (9 photos unannotated), [T25](T25_patient_cases.md)
- **Scenarios**: none; no run or result changes
- **Status**: Done (laptop, 2026-10-28)

---

## 1. Goal

The researcher asked (2026-10-28) to remove from every dataset the 9 included photos that no 23 Oct CVAT export annotates (`roi/crop_summary.json`, `unannotated_records`):

| Photo | Class | Split | Note |
|---|---|---|---|
| P1071425 | cut wound | test | in the cut export with an empty label file; also unannotated in the 30 Sep exports |
| ZZ160063, ZZ240050 | graze wound | validation | boxed on 30 Sep, not on 23 Oct |
| ZZ310855 | contusion | validation | never boxed |
| ZZ230784 | scratch | train | boxed on 30 Sep, not on 23 Oct |
| P6293420 | laceration | train | boxed on 30 Sep, not on 23 Oct |
| ZZ140321, PC235480, IMG_3520 | scratch | train | never boxed |

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Scope | Removed from the committed manifests only; the image files on Raptor are not touched | Asked for the datasets and the tables |
| Split | Not searched again: every other photo keeps its split and fold | No model ever used these photos (`full_scope = matched`: train/validation/test 728/155/149 in every `run_config.json`), so every run stays reproducible and no result changes |
| Groups | Groups 220 (PC235480) and 268 (P6293420) are left without a photo and leave `case_split_assignments.csv` and `cv_folds.csv` | |
| Record | The 9 photos are added to `excluded_photos.csv` with reason `no_annotation`; reports recounted; the split's search settings and score stay as built on 2026-10-23 | |

## 3. Files

| File | Change |
|---|---|
| `src/data/drop_unannotated.py` | New; idempotent |
| `data/relabel_2026_10_23/` | `validated_metadata.csv`, `excluded_photos.csv`, `relabel_report.json`; `splits/` manifest, patient cases, case assignments, CV folds, summary and reports |
| `tests/test_drop_unannotated.py`, `tests/test_relabel.py` | New test; committed counts 1,032 and 9 `no_annotation` |

## 4. Result

- 1,032 photos (728 / 155 / 149), 334 groups (233 / 50 / 51), 391 patients (275 / 61 / 55), 283 CV development groups. The split manifest now holds exactly the photos of the union-ROI manifest, i.e. the photos every model was trained and tested on.
- `included_photos_without_patient_id` 8 → 7 (P1071425 has none; set by hand from the sheet).
- Journal Table 1 and the dataset section updated in the Claude Doc; in the researcher's edited Word file only Table 1's numbers and two notes were changed (backup `… - backup before removing 9 photos.docx`).
- `pytest tests/test_drop_unannotated.py tests/test_relabel.py tests/test_patient_cases.py tests/test_subgroup.py`: 18 passed (Python 3.11, CPU).
