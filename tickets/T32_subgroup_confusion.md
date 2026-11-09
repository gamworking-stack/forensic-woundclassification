# Ticket T32: Confusion Matrices per Subgroup

- **Ticket ID**: `T32`
- **Phase**: Extra analysis of the T23 test results (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T24](T24_subgroups.md) (subgroups, Table 3), [T08](T08_evaluation_module.md) (confusion matrix)
- **Scenarios**: none new; the T23 test predictions of S01–S15 (full image) and S25–S32 (per crop)
- **Status**: Done (laptop, 2026-11-09; results §4)

---

## 1. Goal

The researcher asked (2026-11-09) for confusion matrices of ResNet152, EfficientNet-B0, VGG16 and MobileNetV3-Small divided by age group, race, Monk Skin Tone, setting and body region, as figures in a Word file in the manuscript folder.

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Models | T23 full-image (S01–S15) and per-crop (S25–S32) models of the four families | The models of Table 3 |
| Subgroups | `subgroup.py`'s labels, matching and minimum (10 test photographs); unknowns left out. Race: only Asian reaches 10 (146 of 149) | Same rows as Table 3 |
| Figure | Per attribute and input path: a row per subgroup, a column per family; counts, shaded by the share of the true class (Blues, as `plots.py`); class labels on the outer panels; panel title n and accuracy | Comparable at a glance, readable in print |
| Check | Per family and subgroup, accuracy and image count equal `subgroup_metrics.csv` (T25) | Same images as Table 3 |
| Privacy | Prediction copies in a temporary folder, deleted after the run; outputs hold counts only and the script refuses an output folder inside the repo | T23 §3 |

## 3. Files

| File | Change |
|---|---|
| `src/evaluation/subgroup_confusion.py` | New |
| `tests/test_subgroup.py` | Counts on the synthetic subgroup runs, unknowns left out, one figure per path and attribute |
| `runbook.md` (T23 step 15), `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Command, scope, memo |

Outputs in `../results_figures/relabel_2026_10_23_patient/subgroup_confusion/`: `confusion_<full|crop>_<attribute>.png` and `subgroup_confusion.csv`; Word file `../08_Manuscript/Figures 3-4 (confusion matrices by subgroup).docx` (Figure 3a–e full image, 4a–e per crop).

## 4. Run log and results

- 2026-11-09 (laptop): the 16 T23 test prediction files copied from Raptor, drawn, deleted. All 104 (path × attribute × subgroup × family) accuracies and image counts equal `subgroup_metrics.csv`.
- `pytest tests/test_subgroup.py`: 7 passed (Python 3.11, CPU).
