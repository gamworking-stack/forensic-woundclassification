# Ticket T38: The 19 Nov 2026 Dataset (New Annotation, New Photos, Split by Patient)

- **Ticket ID**: `T38`
- **Phase**: New dataset version for redoing the manuscript experiments (ROADMAP §5)
- **Spec Reference**: [T23](T23_relabel_rerun.md) (layout, environment variables, run order), [T25](T25_patient_cases.md) (patients as cases), [T26](T26_drop_unannotated.md) (photos without a box are dropped)
- **Status**: Dataset prepared on the laptop 2026-11-23; Raptor upload, crops and training pending

---

## 1. Goal

The researcher annotated again in CVAT (19 Nov 2026 exports, `../19 Nov annotation/`, 9 zips: one per class plus `new_cut_19_Nov_added` and `new_scratch_19_Nov_added`) and added photos. Every manuscript experiment is to be redone on this data, with crops from the 19 Nov boxes. This ticket builds the dataset; the experiments then follow the T23 runbook with `scripts/hpc/nov19_env.sh`.

## 2. Source: the researcher's workbook

`../Current dataset T23 (1,032 photos).xlsx`, sheet `Photos`, one row per photo: the T23 photos and the 173 photos new in the 19 Nov exports, with the sheet attributes (filled by the researcher for the new photos on 23 Nov 2026), an anonymous patient number and the split. The workbook is the single source; the 27 Oct sheet is not read again.

## 3. Decisions (researcher, 2026-11-23)

| Item | Decision |
|---|---|
| Boxes and crops | The 19 Nov exports. A photo without a box in them is left out (as in T26): 52 T23 photos and 15 new photos |
| New photos | In when `Inclusion` is TRUE and they have boxes: 155 of 173 (3 Inclusion FALSE, 15 without boxes) |
| Duplicate | ZZ240129 (marked "รูปซ้ำ") is left out |
| Patients | `Patient (anonymous no.)` in the workbook: T23's `patient_case` (1–393) for the T23 photos; 394–486 for new patients (same forensic case number → same patient; the sheet's `patient_id` for older photos; 3 new photos joined existing patients 90, 158, 363) |
| Split | New split of all photos, grouped by patient, with the T02 search (70/15/15, seed 42, 50,000 trials). The patient is the group of the split, CV folds and bootstrap (`pseudonimized_id`) |
| Storage | File names of the new photos hold forensic case numbers, so the dataset's lists stay **outside the repository**: `../datasets/relabel_2026_11_19/` (laptop), `<project>/datasets/relabel_2026_11_19/` (Raptor). Runs: `<project>/experiments_relabel_2026_11_19/`, crops `<project>/derived_data/relabel_2026_11_19/`, results `../results_figures/relabel_2026_11_19/` |
| Methods | Everything else as in T23 |
| Experiments (2026-11-23) | Full image (S01, S03 … S15), per crop (S25–S32), the full-image 5-fold CV, YOLO and the vote (T37), and the CPU analyses (T28, T29, T24/T25, T32). **Not run on T38: per box (S17–S24, T17), CLAHE (T30) and union ROI (S02, S04 … S16).** The union ROI models had already been trained in the main run before this decision; they stay on Raptor but are not reported, and the CV runs full images only (`PATHS=full`, `cv_summary.py --paths full`) |

A photo in no sheet has no sheet case; its `source_pseudonimized_id` is 10000 + its patient number (an integer no sheet case reaches).

## 4. The T38 dataset

| | T23 | T38 |
|---|---|---|
| Photos (train / validation / test) | 1,032 (728 / 155 / 149) | 1,134 (788 / 179 / 167) |
| Split groups | 334 T23 groups (233 / 50 / 51) | 471 patients (330 / 71 / 70) |
| Union crops | 1,032 | 1,134 |
| Per-box crops (train / validation / test) | 1,808 (1,310 / 222 / 276) | 1,925 (1,292 / 333 / 300) |
| CV development groups | 285 (5 folds) | 401 (5 folds of 80–81) |

Photos per class and split:

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| contusion | 119 | 28 | 27 | 174 |
| cut wound | 88 | 22 | 23 | 133 |
| graze wound | 173 | 44 | 38 | 255 |
| gunshot wound | 214 | 47 | 41 | 302 |
| laceration | 85 | 19 | 15 | 119 |
| penetration wound | 38 | 5 | 9 | 52 |
| scratch | 71 | 14 | 14 | 99 |

Of the 1,134 photos, 979 are T23 photos and 155 are new; 98 new photos are not on Raptor yet (`raptor_upload.csv`; the other 57 are older sheet photos already in the raw tree). Four annotated photos match no dataset photo, exactly the four left out above (ZZ240129 and the three Inclusion FALSE photos).

## 5. Files

| File | Change |
|---|---|
| `src/data/prepare_nov19.py` | New: workbook → `validated_metadata.csv`, `workbook_split.csv`, `splits/patient_cases.csv`, `sheet.csv` (subgroup scripts; `patient_id` = the anonymous number), `image_sizes.csv` (EXIF-rotated sizes from the local copies, for the dry-run crops), `raptor_upload.csv`, `excluded_photos.csv`, `prepare_report.json`. Refuses to write inside `data/` |
| `scripts/hpc/nov19_env.sh` | New: `WOUND_DATA_DIR` (outside the repo), `EXPERIMENTS_DIR`, `DERIVED_DIR`, `EXPORTS_DIR`, `RESULTS_DIR` for T38 |
| `scripts/hpc/sync_to_raptor.sh` | Also uploads the 19 Nov exports, the dataset folder and the new photos (into their raw label folder, never replacing a file) |
| `tests/test_nov19.py` | Layout, patients as groups, cases for photos in no sheet, refusals, EXIF size, never inside `data/` |
| `requirements.txt` | `openpyxl` (laptop only, reads the workbook) |
| `src/evaluation/cv_summary.py` | `--paths` (default both): with one path, only `cv_table.csv`, no ROI-full comparison |
| `src/evaluation/verify.py` | CV check reads the paths from `cv_table.csv`; with full only it skips the comparison and its figure (NOTE row) |
| `tests/test_cv.py`, `tests/test_verify.py` | Full-only CV summary and verification |

No other code changes: every step reads `WOUND_DATA_DIR`.

## 6. Acceptance check

1. `create_case_split.py` on the prepared metadata reproduces the workbook's split exactly (checked 2026-11-23: 1,134 of 1,134).
2. Dry-run crops: every photo has a union crop; no unexpected unmatched annotation (checked).
3. `audit_full_vs_roi.py` passes (checked).
4. `pytest tests` passes (laptop 2026-11-23: 103 passed, 4 skipped; the 15 files that need torch run in the Raptor container).
5. On Raptor: both crop jobs end with `ROI comparison PASSED` against the T38 manifests (this also checks the image sizes read on the laptop).

## 7. Run log

- 2026-11-23 (laptop): workbook decisions above; dataset prepared in `../datasets/relabel_2026_11_19/`, split reproduced, union (1,134) and per-box (1,925) dry-run manifests, CV folds, audit passed. `python -m src.detection.boxes` needs torch: run it on Raptor (runbook step 1).
- 2026-11-23 (laptop → Raptor): commit `f8f41e1` (branch `t38-nov19-dataset`) synced; the 98 new photos added to the raw tree (cut 91 → 137, scratch 69 → 121, no file replaced), 19 Nov exports to `annotations/cvat_2026_11_19`, dataset to `datasets/relabel_2026_11_19`. Reference boxes on Raptor: 1,925 on 1,134 photos. Crop jobs 251033 (union, 1,134) and 251034 (per-box, 1,925) both `ROI comparison PASSED` (so the laptop's image sizes are right). Ready for training.
