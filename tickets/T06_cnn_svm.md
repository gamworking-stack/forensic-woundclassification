# Ticket T06: Hybrid CNN-SVM

- **Ticket ID**: `T06`
- **Phase**: Phase 3 (Training Code)
- **Spec Reference**: [specs/phase3_training_code.md](../specs/phase3_training_code.md) §5
- **Scenarios**: S13 (full), S14 (ROI)
- **Status**: In Progress

---

## 1. Goal

Implement `src/training/cnn_svm.py`: extract frozen ImageNet features from **VGG16, ResNet152 and EfficientNet-B0**, concatenate them, tune an SVM on validation macro-F1, refit the chosen SVM on train+validation, and predict the test split once. Output files match the fine-tuned families so T08 evaluates every scenario the same way.

---

## 2. Requirements & Business Logic

1. **Data**: same manifests, matched full scope (D1), leakage and class checks as T05 (`src/training/data.py`).
2. **Features**:
   - `models.feature_extractor(family)` for each extractor, eval mode, no gradients, bf16 autocast on CUDA.
   - Eval transform only (pad to square, resize to 384, ImageNet normalization); no augmentation.
   - Each image is decoded once and shared by all three extractors.
   - Dimensions: VGG16 4,096 (penultimate FC), ResNet152 2,048, EfficientNet-B0 1,280; concatenated 7,424.
   - Saved per extractor and split as `features/<family>_<split>.npz`, and reused on rerun.
   - Test features are extracted only after the SVM has been chosen.
3. **SVM tuning** (train fit, validation score):
   - `StandardScaler` then `SVC(class_weight="balanced")` in one `Pipeline`, so the scaler is fit only on the data the SVM is fit on.
   - Grid: RBF with C ∈ {0.1, 1, 10, 100, 1000} × gamma ∈ {scale, 1e-6, 1e-5, 1e-4, 1e-3}, plus linear with C ∈ {0.001, 0.01, 0.1, 1}: 29 settings. This covers the EXP003 grid, whose best value sat on its edge (C 10, gamma 1e-4).
   - The log warns when the chosen setting is on a grid limit.
   - *Implementation note*: the first Raptor run (jobs 221593/221594, 20-point grid) chose gamma 1e-5, its lower limit, on both paths. Gamma 1e-6 and C 1000 were added before any test metric was computed; selection stays validation-only.
   - Select on validation macro-F1. Ties go to the earlier grid entry (smaller C).
   - Write every grid point to `svm_validation_tuning.csv` (`kernel, C, gamma, validation_macro_f1, validation_balanced_accuracy`).
4. **Final model**:
   - `validation_predictions.csv`: the chosen SVM fit on train only, with `probability=True` (Platt scaling).
   - `test_predictions.csv`: the chosen SVM refit on **train+validation**, with `probability=True`.
   - `predicted_class` is the argmax of the probabilities, as in T05.
   - `random_state = --seed` (42).
5. **Outputs**: `run_config.json` (settings, grid, chosen parameters, feature dimensions, classes, git and environment info), `svm_validation_tuning.csv`, both prediction CSVs, `features/`, `train.log`.
6. **Shared code**: move `Logger`, `resolve_device`, `git_state`, `environment_info` and `prediction_frame` out of `train.py` into `src/training/common.py`, and add the row preparation to `data.py`, so both entry points use one implementation.
7. **Automated Testing** (`tests/test_cnn_svm.py`, synthetic images, CPU, `--no-pretrained`):
   - Writes all outputs. The tuning table has 29 rows. Prediction CSV columns equal T05's, and probabilities sum to 1.
   - Rerun reuses cached features: no new `.npz` files, same tuning table.
   - Test features do not exist when tuning starts (checked through the order in `train.log`).

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/training/cnn_svm.py` | CLI: features, tuning, refit, predictions |
| `src/training/common.py` | Shared logging, device, provenance, prediction CSV |
| `src/training/data.py` | `load_manifest` takes explicit arguments; add `prepare_rows` |
| `src/training/train.py` | Use `common.py` and `prepare_rows` |
| `scripts/slurm/train_single.sbatch` | Accept `FAMILY=cnn_svm`, so every scenario is submitted the same way |
| `tests/test_cnn_svm.py` | Tests above |
| `runbook.md` | Submit commands |

---

## 4. CLI Interface

```bash
python src/training/cnn_svm.py \
  --input-path roi \
  --manifest data/roi/roi_manifest.csv \
  --image-root /project/a260012_FORWCL/derived_data/roi_union_p008 \
  --output-dir /project/a260012_FORWCL/experiments/main/roi/cnn_svm
```

---

## 5. Acceptance Check

1. `pytest tests/` passes, including the existing T05 tests after the refactor.
2. On Raptor, `train_single.sbatch` with `FAMILY=cnn_svm` finishes for `full` and `roi` and writes all outputs. Its best validation macro-F1 on the full path is in the EXP003 range (about 0.55).
3. Tick T06 in `ROADMAP.md` §9.
