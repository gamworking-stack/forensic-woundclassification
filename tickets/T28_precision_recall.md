# Ticket T28: Macro Precision and Recall with Intervals (Manuscript Table 2)

- **Ticket ID**: `T28`
- **Phase**: Manuscript (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T08](T08_evaluation_module.md) (bootstrap), [T25](T25_patient_cases.md) (patient cases)
- **Status**: Done (laptop, 2026-10-28)

## 1. Goal and decision

The researcher asked (2026-10-28) for precision and recall in manuscript Table 2 (full image S01–S15 and per-crop S25–S32). Macro recall equals balanced accuracy and weighted recall equals accuracy (checked on all 16 scenarios), so a separate recall column would repeat a column. Decision (researcher): show macro precision and macro recall, each with a 95% patient-level bootstrap interval, in place of the balanced-accuracy column; a note says recall equals balanced accuracy.

## 2. Method

`src/evaluation/precision_recall.py` draws the same resamples as T08's `bootstrap_draws` (sorted cases, `default_rng(42)`, 2,000 resamples) and scores macro precision and recall over the classes present, unpredicted classes counting 0 (as `compute_metrics`). It refuses to write unless every macro recall and its interval reproduce the saved balanced accuracy and interval, and every macro precision the saved value. The shared bootstrap (`BOOTSTRAP_METRICS`) is not changed, so saved runs and `verify*.py` stay valid.

## 3. Run

- 2026-10-28: the 32 test and validation prediction files of `main/full` and `roi_crop/crop` copied from Raptor to a temporary folder, rewritten per patient (`patient_rescore.py`), scored, deleted. Output `../results_figures/relabel_2026_10_23_patient/precision_recall.csv`; every recall check passed.
- `08_Manuscript/Table 2 (precision recall).docx`: a copy of the researcher's `Table 2.docx` (left unchanged) with the balanced-accuracy and macro-precision columns replaced by macro precision [95% CI] and macro recall [95% CI], and a note under the table.
- `pytest tests/test_precision_recall.py`: 1 passed.
