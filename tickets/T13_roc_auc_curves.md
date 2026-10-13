# Ticket T13: ROC/AUC Curves

- **Ticket ID**: `T13`
- **Phase**: Phase 6 (Figures and Results)
- **Spec Reference**: [specs/phase3_training_code.md](../specs/phase3_training_code.md) §5 (T08 interface: one-vs-rest ROC/AUC)
- **Scenarios**: all (S01–S16)
- **Status**: In Progress (code and tests done; Raptor run pending)

---

## 1. Goal

Draw one-vs-rest ROC curves for the 7 classes, with AUC per class and macro AUC, for every scenario and split. Like the rest of T08, the curves are computed **only from the saved prediction files**, so `evaluate.py` writes them next to the confusion matrix and one evaluation run covers every finished scenario.

---

## 2. Requirements & Business Logic

1. **ROC points** (`metrics.roc_points`):
   - Per class: `sklearn.metrics.roc_curve` on "this class vs the rest" with that class's probability column.
   - A class with no positives or no negatives in the split gets no curve (its AUC is undefined), the same rule as `one_vs_rest_auc`.
   - **Macro average curve**: the class curves linearly interpolated on the union of their false positive rates and averaged. Its AUC is reported as the **mean of the class AUCs**, so the figure, `<split>_metrics.json` (`macro_auc`) and the class-wise report show the same number.
   - Long-format table: `curve` (class name or `macro average`), `fpr`, `tpr`, `threshold` (empty for the macro curve), `auc`.
2. **Figure** (`plots.plot_roc_curves`):
   - One panel: the 7 class curves, the macro average (thick, dark) and the chance diagonal (dotted grey). Each label carries its AUC to 3 decimals.
   - Each class has a fixed colour and dash pattern by its position in the class list, so a class looks the same in all 16 scenarios. The dash patterns keep classes apart in greyscale print and for colour-blind readers.
   - Square axes from 0 to 1, Agg backend, 200 dpi PNG.
3. **Outputs per scenario folder** (added to `evaluate.py`), for `<split>` = validation and test:
   - `<split>_roc_curves.csv`: the points above, so T15 and the thesis can redraw or overlay curves (for example full vs ROI per family) without the models.
   - `<split>_roc_curves.png`.
4. **Automated Testing** (`tests/test_evaluation.py`, CPU):
   - Perfect predictions give AUC 1.0 for every class and the macro curve, and every curve passes through (0, 1).
   - On a hand-built 3-class example, each class AUC equals `class_report`'s AUC and the area under its points; the macro AUC equals `compute_metrics`' `macro_auc`; the macro curve is monotone.
   - A class absent from the split has no curve and raises no error.
   - `--experiments-root` writes both new files for every split.

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/evaluation/metrics.py` | `roc_points` |
| `src/evaluation/plots.py` | `plot_roc_curves`, fixed class colours and dash patterns |
| `src/evaluation/evaluate.py` | Write `<split>_roc_curves.csv` and `.png` per scenario |
| `tests/test_evaluation.py` | Tests above |
| `runbook.md` | New output files |

---

## 4. CLI Interface

No new command: the T08 evaluation run writes the curves.
```bash
python src/evaluation/evaluate.py --experiments-root /project/a260012_FORWCL/experiments/main
```

---

## 5. Acceptance Check

1. `pytest tests/` passes.
2. On Raptor, rerunning the evaluation writes `validation_` and `test_roc_curves.csv` and `.png` for every evaluated scenario, and the test macro AUC in each figure equals `test_macro_auc` in `evaluation_index.csv`.
3. The figures are inspected for S13–S16: 7 class curves, the macro curve and the chance line, with a readable legend.
4. Tick T13 in `ROADMAP.md` §9.
