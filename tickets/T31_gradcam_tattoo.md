# Ticket T31: Grad-CAM of the Tattoo Photos

- **Ticket ID**: `T31`
- **Phase**: Extra analysis (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T14](T14_gradcam.md) (Grad-CAM), [T24](T24_subgroups.md) (tattoo flag), [T23](T23_relabel_rerun.md), [T30](T30_more_contrast.md)
- **Scenarios**: none new; the models of S01–S15 (full image) and S25–S32 (per crop) of T23 and of T30 (CLAHE)
- **Status**: Done (Raptor, 2026-11-09; results §4)

---

## 1. Goal

The researcher asked (2026-11-09) for Grad-CAM of the wounds with a tattoo in the training, validation and test sets, for every experiment behind a table in the manuscript folder (`08_Manuscript/`).

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Photos | Sheet column `tattoo` = 1, included photos of the T23 split: 19 (train 8, validation 8, test 3), 59 crops | T24 definition |
| Experiments | T23 (Tables 2, 2b and 3 use its models) and T30 CLAHE (Table 2c). Table 1 is the data, Table 2b only re-scores the T23 predictions | One set of models per table |
| Models | Full-image model (main run) and per-crop model (S25–S32) of the 7 families with Grad-CAM; CNN-SVM has none (T14 §2); AlexNet ensemble through its representative member (as T14) | As `gradcam_panels.py --all-test` |
| Panels | Each photo's full image with all its boxes, and each crop; overview pages per split, each crop next to its full image with that crop's box | Same layout as the all-test panels |
| Training photos | Explained with the evaluation preprocessing (no augmentation). They have no prediction file, so `matches_prediction_file` is empty; the model saw them in training, so right answers there say little | Validation and test rows are checked against their prediction files |
| Privacy | The photo list is made on the laptop (`subgroup_photos.py`) and holds only manifest paths, split and class; the panels are patient photographs and stay on Raptor | T23 §3, ethics approval |

## 3. Files

| File | Change |
|---|---|
| `src/evaluation/subgroup_photos.py` | New: photos of one sheet subgroup in every split |
| `src/evaluation/gradcam_panels.py` | `--photo-list`: listed photos of any split, filed by split; validation predictions read too |
| `scripts/slurm/gradcam_tattoo.sbatch` | New: tests, then the panels for T23 and T30 (CPU) |
| `tests/test_gradcam.py`, `tests/test_subgroup.py` | Listed photos of every split; the subgroup list |
| `runbook.md` (T23 step 14), `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Commands, scope, memo |

Outputs per experiment in `<run>/roi_crop/gradcam_tattoo/`: `<family>/<split>/<class>/case<case>_<photo>_full.png` and `..._crop<k>.png`, `<family>/<split>_p<n>.png`, and `panels.csv` (split, prediction, right or wrong, agreement with the prediction file, full-image map share inside the boxes).

## 4. Run log and results

- 2026-11-09 (laptop): `subgroup_photos.py` on the 27 Oct corrected sheet: 19 photos (train 8, validation 8, test 3).
- 2026-11-09 (Raptor, job 243722, CPU, 21 min, commit 08fad4d): `pytest tests/test_gradcam.py tests/test_subgroup.py` 20 passed; 546 panels per experiment (7 families × (19 full images + 59 crops)).

Correct (training, validation and test together; training photos were seen in training):

| Family | T23 full | T23 crop | T30 CLAHE full | T30 CLAHE crop |
|---|---|---|---|---|
| EfficientNet-B0 | 16/19 | 52/59 | 15/19 | 50/59 |
| ResNet152 | 18/19 | 52/59 | 18/19 | 54/59 |
| MobileNetV3-Small | 15/19 | 53/59 | 14/19 | 52/59 |
| GoogLeNet | 15/19 | 55/59 | 16/19 | 52/59 |
| VGG16 | 18/19 | 53/59 | 16/19 | 53/59 |
| ViT-B/16 | 19/19 | 53/59 | 19/19 | 52/59 |
| AlexNet ensemble (representative member) | 17/19 | 50/59 | 16/19 | 51/59 |

- Agreement with the prediction files: every test image agrees. 11 of 336 validation checks differ (T23: 6, CLAHE: 5; 9 of them MobileNetV3-Small, one EfficientNet-B0 and one ResNet152), the fp32 (CPU) against bf16 (GPU) effect recorded in T17 §6: the file's top two probabilities are close (e.g. 0.373 / 0.370) or the model is MobileNetV3-Small, which T17 showed to flip under bf16.
