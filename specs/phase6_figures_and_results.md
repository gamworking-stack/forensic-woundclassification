# Phase 6 Specification: Figures and Results

- **Phase**: 6 — Figures and Results
- **Related Tickets**: T13 (ROC/AUC curves), T14 (Grad-CAM), T15 (Summary table and paired full vs ROI), T16 (Final verification)
- **Status**: Completed (as-built spec, written 2026-10-14 after the work; consolidates the tickets, which hold the analysis plans, details and results)
- **Owner**: Forensic Wound Classification Team (Preenapun Saokhieo)

---

## 1. Objectives

Produce every figure and table the proposal asks for from the saved prediction files, answer the main question (does ROI cropping improve classification over the full image?) on the same test cases, and check that every reported number and selection follows from the run files.

---

## 2. Components

| Ticket | Code | Outputs (per run folder on Raptor) |
|---|---|---|
| T13 ROC/AUC | `src/evaluation/metrics.py` (`roc_points`), `plots.py`, run by `evaluate.py` | `<split>_roc_curves.csv` and `.png` per scenario: 7 one-vs-rest curves, macro curve, chance line |
| T14 Grad-CAM | `src/evaluation/gradcam.py` | `gradcam/examples.csv`, `gradcam_results.csv`, `<family>_set1|2.png`; overlays stay on Raptor |
| T15 Summary | `src/evaluation/summarize.py` | `summary/summary_table.csv`, `full_vs_roi.csv`, `full_vs_roi_macro_f1.png` |
| T16 Verification | `src/evaluation/verify.py` | `verification/verification_report.csv` |

The supplementary CV summary (`src/evaluation/cv_summary.py`, T12b) uses the T15 method on the out-of-fold predictions.

---

## 3. Fixed Analysis Rules

- **Metrics** come from prediction files only, never from a live model (T08).
- **Paired comparison** (T15 §2, fixed before any paired result): case-level bootstrap, 2,000 resamples, seed 42, one resample shared by all 16 scenarios; ROI − full per family; two-sided bootstrap p; Holm over 8 families per metric; pooled mean of the 8 differences. Primary macro-F1; secondary balanced accuracy and macro AUC.
- **Grad-CAM** (T14 §2): 7 families (CNN-SVM excluded; AlexNet ensemble through its representative member), the weights that produced the test predictions, one fixed set of 14 test images drawn before any map; full-image maps scored by their share inside the wound box.
- **Not done**: subgroups by `autopsy_status` or skin tone; these columns are not in the metadata used (T15 §2).

---

## 4. Acceptance Checks

1. T13: 64 ROC files; each figure's macro AUC equals `evaluation_index.csv` (T13 §6).
2. T14: 196 maps; predictions equal the prediction files except one documented fp32/bf16 borderline case; figures for the researcher to inspect on Raptor (T14 §6).
3. T15: the summary table equals T08's metrics for 16 of 16 scenarios; paired results recorded (T15 §7).
4. T16: 16 of 16 scenarios and all figures present, test never used for selection, summary numbers equal the per-run files: 313 PASS, 0 FAIL, 2 NOTE (T16 §6).

All met. Open items for the thesis, not for the code: the ethics decision on showing Grad-CAM overlays of patient photographs, and the researcher's inspection of those figures.
