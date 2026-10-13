# Ticket T15: Summary Table and Paired Full vs ROI Comparison

- **Ticket ID**: `T15`
- **Phase**: Phase 6 (Figures and Results)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §1 (main research question), §5 T15; [specs/phase3_training_code.md](../specs/phase3_training_code.md) §5.1 (scenario IDs)
- **Scenarios**: all (S01–S16)
- **Status**: Completed

---

## 1. Goal

Answer the proposal's main question, **does ROI cropping improve classification over the full image**, on the one fixed test set. Write one table of all 16 scenarios, and compare ROI with full image within each family on the **same test cases**, with a paired case-level bootstrap.

---

## 2. Analysis Plan (fixed 2026-10-14, before any paired result was computed)

The unpaired test metrics with their own confidence intervals were already seen (T08 §7). No model or setting changes here; this ticket only fixes how the existing prediction files are compared.

| Item | Decision |
|---|---|
| Unit of resampling | Case (`pseudonimized_id`), with replacement, all images of a case kept together, as in T08 |
| Pairing | Each resample draws one set of cases and scores **both** paths and **all 8 families** on it, so differences are paired by case. Every scenario must have the same test cases (T08 fairness check) |
| Resamples | 2,000, seed 42 (as T08) |
| Primary metric | Test macro-F1 (the model-selection metric) |
| Secondary metrics | Balanced accuracy, macro AUC (one-vs-rest) |
| Effect | ROI − full; positive favours ROI |
| Interval | 95% percentile interval of the bootstrap differences |
| p-value | Two-sided bootstrap: `min(1, 2 × min(P(d ≤ 0), P(d ≥ 0)))` |
| Multiplicity | Holm adjustment over the 8 families, separately per metric |
| Pooled effect | Mean of the 8 family differences, computed on each resample; one interval and p-value |
| Not done | Subgroups by `autopsy_status` or skin tone: these columns are not in the metadata (`data/validated_metadata.csv`) |

---

## 3. Requirements & Business Logic

1. **Inputs**: `<run>/<full|roi>/<family>/test_predictions.csv` and `test_metrics.json` for all 16 scenarios (written by T08). Any missing scenario is an error.
2. **Summary table** (`<run>/summary/summary_table.csv`): one row per scenario in S01–S16 order with scenario ID, family, input path, test images, test cases, accuracy, balanced accuracy, macro precision, macro recall, macro F1 and macro AUC, plus the T08 95% case-bootstrap intervals for accuracy, balanced accuracy, macro F1 and macro AUC. Values are copied from `test_metrics.json`, so they equal T08's.
3. **Paired comparison** (`<run>/summary/full_vs_roi.csv`): one row per family and metric, plus the pooled row: `family, metric, full, roi, difference, ci_low, ci_high, p_value, p_holm, n_cases, resamples`. `full` and `roi` equal the summary table.
4. **Figure** (`<run>/summary/full_vs_roi_macro_f1.png`): forest plot of the ROI − full macro-F1 difference with its 95% interval per family and pooled, a zero line, and the value printed per row. Greyscale-safe.
5. **Automated Testing** (`tests/test_summary.py`, CPU, hand-made prediction files):
   - Identical full and ROI predictions give a difference of 0 and p = 1.
   - ROI perfect and full partly wrong gives a positive difference whose interval excludes 0.
   - Holm adjustment matches a hand-computed example.
   - A scenario with different test cases, or a missing scenario, raises.
   - Summary table has 16 rows in S01–S16 order and equals the metrics JSON; the figure is written.

---

## 4. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/evaluation/summarize.py` | CLI: summary table, paired bootstrap, figure |
| `src/evaluation/plots.py` | `plot_full_vs_roi` |
| `tests/test_summary.py` | Tests above |
| `runbook.md` | Command and outputs |

---

## 5. CLI Interface

```bash
python src/evaluation/summarize.py --experiments-root /project/a260012_FORWCL/experiments/main
```

---

## 6. Acceptance Check

1. `pytest tests/` passes.
2. On Raptor, after `evaluate.py`, the command writes the three files; the summary table's macro F1 equals `evaluation_index.csv` for all 16 scenarios.
3. The figure is inspected: 8 family rows and the pooled row, readable labels.
4. Record the paired results here and tick T15 in `ROADMAP.md` §9.

---

## 7. HPC Run (2026-10-14)

`summarize.py --experiments-root /project/a260012_FORWCL/experiments/main`: 16 scenarios, 62 test cases (157 images), 2,000 paired resamples. `pytest tests/test_summary.py` passed in the Raptor container (6 passed). The summary table's macro F1 equals `evaluation_index.csv` for all 16 scenarios; the figure was inspected (8 family rows, pooled row, labels readable).

**Primary: test macro-F1, ROI − full**

| Family | Full | ROI | Difference [95% paired CI] | p | Holm p |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0.733 | 0.742 | +0.009 [−0.128, +0.136] | 0.903 | 1.000 |
| ResNet152 | 0.783 | 0.726 | −0.058 [−0.163, +0.049] | 0.263 | 1.000 |
| MobileNetV3-Small | 0.653 | 0.727 | +0.074 [−0.061, +0.183] | 0.309 | 1.000 |
| GoogLeNet | 0.705 | 0.736 | +0.031 [−0.099, +0.160] | 0.661 | 1.000 |
| VGG16 | 0.699 | 0.715 | +0.017 [−0.123, +0.132] | 0.848 | 1.000 |
| ViT-B/16 | 0.754 | 0.761 | +0.007 [−0.091, +0.116] | 0.869 | 1.000 |
| CNN-SVM | 0.585 | 0.725 | +0.140 [+0.034, +0.249] | 0.007 | 0.056 |
| AlexNet ensemble | 0.662 | 0.728 | +0.066 [−0.048, +0.161] | 0.246 | 1.000 |
| **All families (mean)** | 0.697 | 0.732 | **+0.036 [−0.031, +0.091]** | 0.288 | – |

**Secondary (pooled)**: balanced accuracy +0.040 [−0.043, +0.109], p = 0.347; macro AUC +0.021 [−0.011, +0.059], p = 0.237. Unadjusted p < 0.05 only for CNN-SVM balanced accuracy (0.047) and macro AUC (0.013) and MobileNetV3-Small macro AUC (0.043); none survives Holm (0.376, 0.104, 0.301).

**Reading**
- ROI is ahead in 7 of 8 families and pooled, but no family difference survives the Holm adjustment and the pooled interval includes 0: no clear evidence that ROI cropping improves macro-F1 on this test set.
- CNN-SVM is the only family whose interval excludes 0 before adjustment (Holm p = 0.056). Its features are frozen ImageNet features of the whole frame, which may explain why removing background helps it most; this is a hypothesis, not tested here.
- Per-family paired intervals are about ±0.13 wide, so differences below about 0.1 cannot be resolved with 62 test cases; the pooled interval is compatible with a small loss (−0.03) up to a moderate gain (+0.09). Report this as a limitation.
