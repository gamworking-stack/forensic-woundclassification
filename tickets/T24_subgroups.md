# Ticket T24: Test Performance by Age, Race/Ethnicity and Skin Tone (T23 data)

- **Ticket ID**: `T24`
- **Phase**: Extra analysis of the T23 rerun, outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T23](T23_relabel_rerun.md) (data, split, runs); [T08](T08_evaluation_module.md) (metrics, case bootstrap); [T15](T15_summary_and_full_vs_roi.md) §2 (subgroups were "not done": the columns were not in the metadata)
- **Scenarios**: none new; the T23 test predictions of S01–S32 scored per subgroup
- **Status**: Age, race/ethnicity and skin tone done (run on the laptop 2026-10-26; results in §6). Setting, body region and tattoo added and run 2026-10-28 (results §7)

---

## 1. Goal

Describe how test performance varies with the patient's **age** and **race/ethnicity** and the wound's **skin tone**, using the researcher's 26 Oct 2026 sheet (`../thesis_data_26_oct.csv`). It has the same photos, labels and `Inclusion` as the 23 Oct sheet, plus updated `skin_tone` and `body_site` values and a new `tattoo` column. No model is trained and no new test prediction is made.

---

## 2. Decisions (2026-10-26, after every test result had been seen)

Post hoc and **descriptive**: decided after the T23 test results had been seen.

| Item | Decision | Reason |
|---|---|---|
| Data | T23 test set: 149 photos, 51 groups (the full-image photos matched to ROI). Demographics from `../thesis_data_26_oct.csv`, `Inclusion = TRUE` rows, joined on (sheet case ID, filename) | The T23 models; the newest sheet |
| Age groups | <18, 18–39, 40–59, 60+. Age `0`, `-` or blank = unknown (the 0s come with race `0`) | Broad adult bands; the test set has no one under 18 |
| Race/ethnicity | Asian, European; `-`, `0`, `?`, `ไม่พบ ID` = unknown | Sheet values |
| Skin tone | Monk Skin Tone (ROADMAP §1) merged into MST 1–3, MST 4, MST 5–10; blank = unknown | The test set has MST 2–6 only; MST 4 is half of it |
| Setting, body region, tattoo (added 2026-10-28) | Per photo: `autopsy` 1 = autopsy (deceased), 0 = living (coding confirmed by the researcher 2026-10-28). `body_site` merged into head and neck (scalp, face, ear, neck), upper limb (shoulder, armpit, arm, elbow, wrist, hand, finger), trunk (chest, abdomen, back), lower limb (leg, knee, foot) and multiple sites. `tattoo` 1 / 0. Anything else (`unknown`, `N/A`, blank) = unknown | The other columns of the sheet, as in the journal characteristics table |
| Scored subgroups | Known subgroups with ≥ 10 test photos. Smaller and unknown subgroups are counted only | Fewer than 10 photos give no usable estimate |
| Scenarios | Full image and union ROI (S01–S16), per-box ROI at photo level (S17–S24), per-crop ROI (S25–S32, one row per crop) | Every T23 test result |
| Metrics | Accuracy, balanced accuracy, macro-F1 (over the classes present in the subgroup, as T08 defines them). No AUC | AUC is unstable with 2–5 classes in 7–28 groups |
| Intervals | One case-level bootstrap of the whole test set per input path (2,000 resamples, seed 42); each subgroup is rescored on its own rows in every resample. The pooled row is the mean over the 8 families, interval from the per-resample means (as T15) | Cases stay together; families share the resamples |
| Comparison | Each subgroup minus the largest subgroup of its attribute (18–39; MST 4; autopsy; the largest body region), with interval and two-sided bootstrap p. **No multiplicity correction** | Descriptive only |
| Privacy | `patient_id` and per-photo demographics are never written. Outputs go outside the repo, and the script refuses an output folder inside it. The test prediction files were copied from Raptor to a temporary folder on the laptop; the sheet was not uploaded | T23 §3 |

---

## 3. Requirements

1. `src/evaluation/subgroup.py --sheet <csv> --main-root <exp>/main --box-root <exp>/roi_box [--crop-root <exp>/roi_crop] --output-dir <outside repo>` writes
   - `subgroup_counts.csv`: photos, groups and class mix per subgroup, unknowns included;
   - `subgroup_metrics.csv`: per scenario and pooled, per scored subgroup: rows, photos, groups, the three metrics with intervals;
   - `subgroup_gaps.csv`: each scored subgroup minus its attribute's largest, per scenario and pooled;
   - `subgroup_class_accuracy.csv`: accuracy within each wound class (its recall) per subgroup, every subgroup however small, per family and pooled, with row and photo counts (added 2026-10-26);
   - `subgroup_accuracy.png`, `subgroup_macro_f1.png`: pooled score per subgroup and input path.
2. It fails if a test photo is not an included sheet row, if a prediction row matches no test photo or has another case or class, or if the families of a path were scored on different test sets.
3. Tests (`tests/test_subgroup.py`, CPU, hand-made files): binning and unknowns, the fast metrics equal `summary_metrics`, subgroup scores and gaps on a known example, setting, body region and tattoo coding, excluded and padding sheet rows ignored, output inside the repo refused, a photo missing from the sheet is an error.

---

## 4. Files

| File | Change |
|---|---|
| `src/evaluation/subgroup.py` | New |
| `tests/test_subgroup.py` | Tests |
| `runbook.md`, `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Command, scope, memo |

---

## 5. Run log

- 2026-10-26 (laptop): the 40 T23 `test_predictions.csv` files (main full/roi, `roi_box` box and photo, `roi_crop`) were copied from Raptor to a temporary folder and the script was run there (16 s). Outputs: `../results_figures/relabel_2026_10_23/subgroup/`.
- `pytest tests/test_subgroup.py`: 5 passed (Python 3.11, CPU). Also passing: `test_single_box`, `test_summary`, `test_relabel`, `test_evaluation`, `test_top_crop` (35 tests).
- 2026-10-27 (laptop): checked against the researcher's 27 Oct sheet (`../thesis_data_27_oct.csv`). Compared with 26 Oct: the same 1,113 rows, labels, case and patient IDs, age and race; `Inclusion` set to FALSE for the 7 `imprint` photos (already excluded, T23 §3); skin tone filled for 3 photos, `wound_id` for 1, `tattoo` for 8, `body_site` for 357 (none of the last three is used here). All 4 photos with a changed skin tone or `wound_id` are in the training split. Every test photo gets the same age, race/ethnicity and skin-tone group as before, and `subgroup_counts.csv` from the 27 Oct sheet is identical to the 26 Oct one. With the predictions, groups and seed unchanged, the §6 results hold for the 27 Oct sheet; the full rescore was not rerun (Raptor unreachable, prediction copies deleted after the 26 Oct run). `pytest` on `test_subgroup`, `test_relabel`, `test_single_box`, `test_summary`, `test_top_crop`: 25 passed (Python 3.11, CPU).

---

## 6. Results (test: 149 photos, 51 groups)

Test photos per subgroup (groups in brackets):

| Attribute | Scored | Counted only |
|---|---|---|
| Age | 18–39: 64 (28); 40–59: 49 (14); 60+: 24 (7) | unknown: 12 (3); <18: 0 |
| Race/ethnicity | Asian: 146 (49) | European, unknown: < 10 |
| Skin tone | MST 1–3: 34 (19); MST 4: 75 (36); MST 5–10: 40 (24) | – |

Pooled over the 8 families (mean), with 95% case-bootstrap intervals:

| Subgroup | Full image acc / macro-F1 | Union ROI acc / macro-F1 | Per-box ROI acc / macro-F1 | Per-crop acc / macro-F1 |
|---|---|---|---|---|
| Age 18–39 | 0.807 / 0.660 | 0.795 / 0.627 | 0.842 / 0.650 | 0.842 / 0.644 |
| Age 40–59 | 0.732 / 0.657 | 0.842 / 0.795 | 0.857 / 0.816 | 0.743 / 0.699 |
| Age 60+ | 0.807 / 0.778 | 0.891 / 0.883 | 0.901 / 0.898 | 0.838 / 0.832 |
| Asian | 0.789 / 0.682 | 0.836 / 0.748 | 0.861 / 0.760 | 0.805 / 0.704 |
| MST 1–3 | 0.702 / 0.643 | 0.779 / 0.706 | 0.805 / 0.703 | 0.670 / 0.639 |
| MST 4 | 0.815 / 0.688 | 0.852 / 0.773 | 0.883 / 0.782 | 0.839 / 0.746 |
| MST 5–10 | 0.822 / 0.694 | 0.866 / 0.768 | 0.878 / 0.742 | 0.874 / 0.671 |

Differences from the largest subgroup (pooled, accuracy):

| Comparison | Full image | Union ROI | Per-box ROI | Per-crop |
|---|---|---|---|---|
| 40–59 − 18–39 | −0.074 [−0.207, +0.079] | +0.047 [−0.075, +0.171] | +0.015 [−0.106, +0.131] | −0.100 [−0.282, +0.077] |
| 60+ − 18–39 | +0.001 [−0.120, +0.103] | +0.096 [+0.004, +0.198] | +0.059 [−0.028, +0.172] | −0.004 [−0.097, +0.117] |
| MST 1–3 − MST 4 | −0.113 [−0.249, +0.036] | −0.072 [−0.202, +0.064] | −0.078 [−0.220, +0.066] | −0.169 [−0.365, +0.050] |
| MST 5–10 − MST 4 | +0.007 [−0.133, +0.141] | +0.014 [−0.099, +0.120] | −0.005 [−0.110, +0.093] | +0.034 [−0.053, +0.109] |

- **Race/ethnicity cannot be assessed**: 146 of the 149 test photos are of Asian patients.
- **Skin tone**: accuracy on MST 1–3 is lower than on MST 4 for every input (−0.07 to −0.17), but every interval includes 0. The balanced-accuracy gap is smaller (full image −0.015, union −0.075, per-box −0.068, per-crop −0.080). MST 1–3 holds 7 of the 13 cut-wound test photos, a hard class, so class mix may explain part of the gap. MST 5–10 is no different from MST 4.
- **Age**: with ROI, patients 60+ score higher than 18–39 (union ROI accuracy +0.096 [+0.004, +0.198]; macro-F1 +0.257 [+0.090, +0.365] union, +0.248 [+0.019, +0.352] per-box). But the 60+ group is 7 groups, has no scratch or penetration wound and only 2 gunshot photos, and macro-F1 is averaged over its 5 present classes. This is a class-mix and sample-size effect, not evidence that age matters. The full image shows no age difference.
- **Per family**: 6 of the 128 family-level accuracy comparisons have p < 0.05 (4 inputs × 8 families × 4 comparisons), about the number expected by chance.
- **ROI is above the full image in almost every subgroup**: pooled accuracy of union and per-box ROI beats the full image in every scored subgroup except union ROI for 18–39 (−0.012). In the 18–39 group, macro-F1 is below the full image for both crop designs.
### Accuracy within each wound class by skin tone (added 2026-10-26)

Pooled over the 8 families; test photos (groups) per cell in brackets. Per-crop rows are crops and are left out here.

| Class | MST 1–3: full / union / per-box | MST 4: full / union / per-box | MST 5–10: full / union / per-box |
|---|---|---|---|
| Gunshot wound | 0.93 / 0.98 / 1.00 (7, 5) | 0.92 / 0.92 / 0.92 (21, 11) | 0.89 / 0.93 / 0.92 (13, 7) |
| Graze wound | 0.84 / 1.00 / 0.97 (4, 3) | 0.88 / 0.88 / 0.96 (22, 13) | 0.77 / 0.87 / 0.92 (14, 11) |
| Contusion | 0.75 / 0.75 / 0.75 (8, 6) | 0.88 / 0.96 / 0.94 (10, 7) | 0.93 / 0.80 / 0.83 (5, 5) |
| Laceration | 0.63 / 0.69 / 0.88 (2, 2) | 0.78 / 0.71 / 0.83 (10, 9) | 0.72 / 0.78 / 0.75 (4, 4) |
| Cut wound | **0.46 / 0.57 / 0.63 (7, 3)** | 0.92 / 0.83 / 0.71 (3, 2) | 0.96 / 0.96 / 1.00 (3, 2) |
| Scratch | 0.55 / 0.80 / 0.88 (5, 2) | 0.41 / 0.70 / 0.71 (7, 5) | – (0) |
| Penetration wound | 0.75 / 0.25 / 0.00 (1, 1) | 0.25 / 0.56 / 0.50 (2, 2) | 0.13 / 0.38 / 0.13 (1, 1) |

- Most of the MST 1–3 accuracy gap comes from **cut wounds**. MST 1–3 holds 7 of the 13 cut-wound test photos, and they are classified correctly less often (0.46 full image; per-crop 0.36 over 23 crops) than the 3 cut wounds in each of the other groups. But those 7 photos come from only 3 groups, so this may be about those patients' wounds rather than skin tone.
- Contusion is also lower on MST 1–3 (0.75 for all three inputs, vs 0.88–0.96 on MST 4; 6 groups). Gunshot and graze wounds are not lower on lighter skin.
- Penetration, laceration (MST 1–3, 5–10) and scratch cells hold 1–5 photos: not interpretable.

- **Limits**: post hoc, small subgroups (7–36 groups), no multiplicity correction, subgroups differ in class mix, and a single hospital with an almost entirely Asian population. Treat these results as description, not as a fairness evaluation.

---

## 7. Setting, body region and tattoo (added 2026-10-28)

The researcher asked for the subgroup analysis on every feature of the journal characteristics table. `subgroup.py` now also scores setting (living or autopsy), body region and tattoo, coded as in §2; the three original attributes are unchanged.

**Sheet.** The 27 Oct file the researcher saved at 21:52 (`../thesis_data_27_oct.csv`) is the split manifest with the sheet's columns looked up beside it. It has no `patient_id`, and the lookup was by file name alone, so the three file names listed twice (ZZ260113, P5270099, P6293420) carry another photo's values. `../thesis_data_27_oct_corrected.csv` (outside the repo, holds `patient_id`) was built from the 26 Oct sheet with the 27 Oct values copied in by (sheet case, file name, label); those three rows keep their own 26 Oct values, with blank tattoo as 0 and blank body site as `unknown` like the rest of the 27 Oct file, and the 7 imprint photos are `Inclusion = FALSE`. `prepare_relabel.py` on it gives the T23 metadata again (only `wound_id` differs), and every test photo keeps its §6 age, race and skin-tone group.

Test photos per new subgroup (149 matched to ROI; cases in brackets): autopsy 129 (48), living 18 (2), setting unknown 2 (1); head and neck 27 (18), upper limb 33 (21), trunk 23 (15), lower limb 30 (18), region unknown 36 (21); no tattoo 146 (49), tattoo 3 (2). Living is scored; its 2 T23 groups hold 6 patients (T25). Tattoo is counted only.

In the corrected sheet, two training photos have autopsy 0 while the same patient's other photos have 1 (P9026335, case 57; ZZ102649, case 183); they do not touch the test set. Shown to the researcher.

- 2026-10-28 (laptop): `pytest tests/test_subgroup.py` 5 passed (Python 3.11, CPU). The 32 test prediction files copied from Raptor to a temporary folder, scored (13 s) and deleted; outputs replace those in `../results_figures/relabel_2026_10_23/subgroup/`. Age, race/ethnicity and skin tone reproduce §6 exactly (same predictions, groups and seed).

**Results** (mean of the 8 families, 95% case-bootstrap intervals; difference from the largest subgroup):

| Comparison | Full image accuracy | Union ROI | Per-box ROI | Per-crop |
|---|---|---|---|---|
| Living − autopsy | −0.196 [−0.308, −0.079] | −0.083 [−0.171, +0.013] | −0.057 [−0.136, +0.028] | −0.066 [−0.192, +0.050] |
| Head and neck − upper limb | +0.085 [−0.061, +0.210] | −0.003 [−0.132, +0.125] | −0.000 [−0.108, +0.113] | −0.032 [−0.147, +0.086] |
| Trunk − upper limb | +0.059 [−0.120, +0.204] | +0.029 [−0.153, +0.168] | −0.016 [−0.201, +0.125] | −0.036 [−0.156, +0.039] |
| Lower limb − upper limb | +0.090 [−0.032, +0.200] | −0.024 [−0.174, +0.093] | −0.000 [−0.118, +0.115] | −0.075 [−0.216, +0.050] |

- **Setting**: the full-image gap (also macro-F1 −0.130 [−0.210, −0.020]; balanced accuracy −0.121 [−0.183, +0.048]) rests on 2 T23 groups (6 patients; T25): one patient's 7 scratches (accuracy 0.41; autopsy scratches 0.55) and another's 2 grazes (0.00); living contusions score 0.91 (autopsy 0.81). With 2 clusters the bootstrap understates the uncertainty; resampling the 6 patients (T25) gives −0.196 [−0.382, +0.093]. Class mix and sample size, not evidence about the setting; with ROI input the gap is −0.06 to −0.08 and every interval includes 0.
- **Body region**: no difference for any input; every interval includes 0.
- **Tattoo**: 3 test photos with a tattoo, not scored.
- **Per family**: 14 of the 128 family-level accuracy comparisons of setting and body region have p < 0.05; 9 of them are living − autopsy (6 on full images).
- Journal Table 5 (Claude Doc) reports full image and per-crop for all six attributes.
