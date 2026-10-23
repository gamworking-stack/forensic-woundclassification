# Ticket T23: Rerun Every Experiment on the 23 Oct 2026 Labels and Annotations

- **Ticket ID**: `T23`
- **Phase**: Rerun of Experiments 1–4 and their supplementary analyses on a new dataset version (ROADMAP §5 "Extra experiments")
- **Spec Reference**: every earlier ticket. T01, T02, T03 and T12b (CV folds) are redone on the new sheet and the new CVAT exports. T19 §2.1 (reference boxes) is redone too. Training, evaluation and verification are unchanged.
- **Scenarios**: S01–S32 and the T19 arms again, under the same IDs, in their own experiments folder
- **Status**: Manifests built and committed 2026-10-23. HPC runs not started: Raptor was unreachable from the laptop (off the CMU network) when the manifests were built.

---

## 1. Goal

Rerun every experiment with the researcher's updated sheet `../thesis_data_23_oct.csv` and updated CVAT annotations `../23 oct updated annotation CVAT/`. Photos marked `Inclusion = FALSE` are excluded. The results are saved separately: the original manifests, runs, crops and fetched results are not touched.

---

## 2. What changed in the sheet and the annotations

| | 30 Sep sheet (original study) | 23 Oct sheet |
|---|---|---|
| Photos | 1,106 | 1,113 (7 new photos, all `imprint`) |
| `Inclusion` | – | 1,048 TRUE, 65 FALSE |
| Labels | 7 classes | 7 classes + `imprint`; 2 photos changed from gsw to penetration wound (case 59: PB098804, PB098817) |
| `pseudonimized_id` | 421 cases | renumbered and regrouped from case 316 on |
| `patient_id` | – | new column; disagrees with `pseudonimized_id` in 37 cases; 8 rows `n/a`, 7 blank |

The annotations are 7 new CVAT exports, one per class (`<class>_22_Oct.zip`). Six are YOLO 1.1 as before; laceration is CVAT for images 1.1 (`annotations.xml`).

| | 30 Sep exports | 23 Oct exports |
|---|---|---|
| Photos / boxes | 1,104 / 1,677 | 1,100 / 1,972 |
| Photos with the same boxes as before | – | 536 of the 1,100 (the others redrawn, split into more boxes, or removed) |
| Photos no longer annotated | – | 4 (graze ZZ160063, ZZ240050; laceration P6293420; scratch ZZ230784) |
| Rotated boxes | – | 8, all laceration (XML keeps the rotation) |

The 2 relabelled photos are still annotated in the gsw export.

Two things were found in the sheet while preparing the data, and shown to the researcher:

- From case 316 on, the new `pseudonimized_id` puts photos of different patients and dates in one case (e.g. new case 324: photos from 28 Jul, 1 Sep and 17 Sep). It also splits one patient's same-day series over several cases (P7190311–P7190412 over cases 319–321).
- The old 30 Sep case IDs agree with `patient_id`: no old case holds two patients, and only old cases 315 and 317 share one. Under the frozen split, 17 included new cases would have photos in more than one of train, validation and test.

---

## 3. Decisions (researcher, 2026-10-23)

| Item | Decision | Reason |
|---|---|---|
| Exclusions | `Inclusion = FALSE` (65 photos) and `imprint` (7 photos: not one of the 7 classes, no CVAT boxes) | Keeps the 7-class design |
| Split | New split with the T02 search (seed 42, 50,000 trials, 70/15/15). The original split stays frozen for the original study | The regrouped cases leak under the frozen split |
| Grouping | Photos sharing the **new** `pseudonimized_id` **or** the same `patient_id` form one group (connected components); `n/a` and blank patient IDs join nothing. The group becomes `pseudonimized_id`, so the split, CV folds, leakage checks and case bootstrap all use it; the sheet's own ID is kept as `source_pseudonimized_id` | Chosen over grouping by the old case IDs + `patient_id` (396 groups, largest 14 photos), knowing the new IDs look misaligned from case 316 on; still leak-safe, but some groups join unrelated patients (largest 36 photos) |
| Relabelled photos | New class for training and scoring; the raw image and CVAT boxes stay filed under the old label (`source_label`) | The images on Raptor and the annotations were not moved |
| Boxes | The 23 Oct CVAT exports (researcher, 2026-10-23). A rotated box is cropped to the axis-aligned rectangle that encloses it | Updated annotations. The enclosing rectangle keeps the whole marked wound in the crop (YOLO 1.1 cannot store a rotation; the YOLO exports may hold rotated boxes whose rotation CVAT dropped) |
| Methods | Everything else as in the original study: families, lr grid, budgets, seeds, metrics, bootstrap, Holm, T19 gate and margin | Only the data changes |
| Storage | `data/relabel_2026_10_23/`, `<project>/experiments_relabel_2026_10_23/`, `<project>/derived_data/relabel_2026_10_23/`, `../results_figures/relabel_2026_10_23/` | Results kept apart from the original study |

`patient_id` and the demographic columns are never written to the repository.

---

## 4. The T23 dataset (committed manifests)

| | Original | T23 |
|---|---|---|
| Photos / groups | 1,106 / 421 | 1,041 / 336 (from 392 sheet cases; 19 groups join several) |
| Split, groups | 294 / 62 / 62 | 235 / 50 / 51 |
| Split, photos | 779 / 164 / 157 (matched to ROI) | 733 / 158 / 150 (728 / 155 / 149 with an ROI crop) |
| Union crops | 1,100 | 1,032 (728 / 155 / 149; 9 photos unannotated) |
| Per-box crops | 1,667 (1,177 / 277 / 213) | 1,808 (1,310 / 222 / 276) |
| CV development groups | 356 | 285 (5 folds of 57) |
| T19 oversize share / most boxes per photo (train) | 0.443 / 35 | 0.611 / 53 |

Photos per class and split (full image):

| Class | Train | Validation | Test |
|---|---|---|---|
| contusion | 128 | 28 | 23 |
| cut wound | 59 | 14 | 14 |
| graze wound | 165 | 35 | 40 |
| gunshot wound | 222 | 44 | 41 |
| laceration | 82 | 23 | 16 |
| penetration wound | 33 | 8 | 4 |
| scratch | 44 | 6 | 12 |

Penetration wound has only 4 test photos (7 in the original split), so its per-class test scores will be very noisy.

---

## 5. Code changes

| File | Change |
|---|---|
| `src/data/prepare_relabel.py` | New: sheet → exclusions, groups, `source_label`, T01 validation; writes `validated_metadata.csv`, `excluded_photos.csv`, both reports |
| `src/data/paths.py` | New: `DATA_DIR` from `WOUND_DATA_DIR` (default `data`); every script's `data/...` default now reads it |
| `src/data/create_roi_dataset.py` | `find_source_image` and `manifest_key` use `source_label` when present. Crop rows and folders take the class from the manifest row, not the annotation (the same for the original data). `parse_cvat_xml_export` reads CVAT XML exports (rotated box → enclosing rectangle); `parse_export` picks XML or YOLO per zip |
| `src/detection/boxes.py` | Reads either export format through `parse_export` |
| `src/detection/common.py` | `photo_table` keeps `source_label` |
| `src/evaluation/verify_auto.py` | Boxes per split are checked against the per-box manifest, not the constants 1,177 / 277 / 213 |
| `scripts/slurm/*.sbatch`, `scripts/hpc/submit_*.sh`, `fetch_results.sh` | `WOUND_DATA_DIR`, `EXPERIMENTS_DIR`, `DERIVED_DIR`, `RESULTS_DIR`, all defaulting to the original paths |
| `scripts/hpc/relabel_env.sh` | New: sets them for T23, plus `EXPORTS_DIR` (the 23 Oct exports on Raptor) |
| `scripts/hpc/sync_to_raptor.sh` | Also uploads the 23 Oct exports to `<project>/annotations/cvat_2026_10_23/` |
| `tests/test_relabel.py` | Grouping, exclusions, relabel matching, relabelled crops, CVAT XML and rotated boxes, `WOUND_DATA_DIR`, committed manifests |

With none of the variables set, every script reads and writes exactly what it did before.

---

## 6. Acceptance check

1. `pytest tests` passes. Locally on 2026-10-23 (Python 3.11, CPU); also in the Raptor container through the T03 job.
2. Both T03 crop jobs end with `ROI comparison PASSED` against the T23 manifests.
3. Every scenario of S01–S32 is finished in `experiments_relabel_2026_10_23/`.
4. `verify.py`, `verify_box.py`, `verify_crop.py` and (if any arm passes its gate) `verify_auto.py` report 0 FAIL.
5. The results are recorded here, next to the original ones, without replacing them.

Run order and commands: runbook, "Relabelled rerun (T23)".

---

## 7. Run log

- 2026-10-23: manifests built on the laptop with the 23 Oct annotations and committed (§4). Raptor not reachable (`nc` timeout, off the CMU network); nothing submitted yet.
