# Ticket T39: Six Classes, Cut and Penetration Wounds Merged into Sharp Injury (19 Nov 2026 Data)

- **Ticket ID**: `T39`
- **Phase**: Variant of T38 for the manuscript (ROADMAP §5)
- **Spec Reference**: [T38](T38_nov19_dataset.md) (photos, annotation, patients, experiments), [T23](T23_relabel_rerun.md) (run order)
- **Status**: Dataset prepared 2026-11-23; Raptor runs queued behind T38

---

## 1. Goal

The researcher asked (2026-11-23) to redo everything with cut wounds and penetration wounds combined into one class, **sharp injury**: 6 classes instead of 7. Photos, patients, exclusions, boxes and experiments are those of T38; T38 (7 classes) is finished first and kept.

## 2. Decisions (researcher, 2026-11-23)

| Item | Decision |
|---|---|
| Classes | contusion, graze wound, gunshot wound, laceration, scratch, **sharp injury** (= cut wound + penetration wound, including the 2 photos of case 59 relabelled from gsw to penetration) |
| Split | **New split** for the 6 classes: the T02 search again (70/15/15, seed 42, 50,000 trials), grouped by the anonymous patient number. Patients may sit in another set than in T38, so T38 and T39 are not compared on the same test photos |
| Order | T38's remaining jobs (CV, YOLO) finish first; T39 training starts after them |
| Experiments | As T38: full image, per crop, full-image CV, Grad-CAM, YOLO 4Y and the vote, the CPU analyses and the manuscript tables. Not run: per box, CLAHE, union ROI |
| Storage | `../datasets/relabel_2026_11_19_sharp/` (outside the repo, like T38), `<project>/experiments_relabel_2026_11_19_sharp/`, `<project>/derived_data/relabel_2026_11_19_sharp/`, `../results_figures/relabel_2026_11_19_sharp/`; photos and 19 Nov exports are T38's |

The merge is applied to the training class only: `source_label` keeps the annotation's class (cut, penetrating wound (sharp&blunt), gsw), so the crops still find their CVAT boxes and the raw photos their folders.

## 3. The T39 dataset

| | T38 (7 classes) | T39 (6 classes) |
|---|---|---|
| Photos (train / validation / test) | 1,134 (788 / 179 / 167) | 1,134 (799 / 174 / 161) |
| Patients | 471 (330 / 71 / 70) | 471 (330 / 71 / 70) |
| Union crops | 1,134 | 1,134 |
| Per-box crops (train / validation / test) | 1,925 (1,292 / 333 / 300) | 1,925 (1,374 / 257 / 294) |
| CV development patients | 401 (5 folds) | 401 (5 folds of 80–81; every fold holds all 6 classes) |

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| contusion | 118 | 29 | 27 | 174 |
| graze wound | 181 | 39 | 35 | 255 |
| gunshot wound | 212 | 50 | 40 | 302 |
| laceration | 84 | 17 | 18 | 119 |
| scratch | 75 | 12 | 12 | 99 |
| sharp injury | 129 | 27 | 29 | 185 |

## 4. Files

| File | Change |
|---|---|
| `src/data/paths.py` | `EXPECTED_CLASSES` from `WOUND_N_CLASSES` (default 7) |
| `src/training/data.py`, `src/data/create_cv_folds.py`, `src/data/create_case_split.py` | Use it instead of the constant 7 |
| `src/data/prepare_nov19.py` | `--merge ... --into ...`; `--patient-cases-from-split` (the merged version is split again) |
| `src/evaluation/subgroup_confusion.py` | Short label SHP for sharp injury |
| `scripts/hpc/sharp_env.sh` | New: T39 folders and `WOUND_N_CLASSES=6` |
| `scripts/hpc/sync_to_raptor.sh` | Also uploads the T39 dataset folder |
| `tests/conftest.py` | Removes `WOUND_N_CLASSES` before the tests import anything (they build 7-class data) |
| `tests/test_nov19.py` | The merge |

With `WOUND_N_CLASSES` unset, every script behaves as before.

## 5. Run log

- 2026-11-23 (laptop): dataset prepared with `--merge "cut wound" "penetration wound" --into "sharp injury"`, new split, patient cases, union (1,134) and per-box (1,925) dry-run crops, CV folds, audit passed.
