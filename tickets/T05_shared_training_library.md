# Ticket T05: Shared Training Library

- **Ticket ID**: `T05`
- **Phase**: Phase 3 (Training Code)
- **Spec Reference**: [specs/phase3_training_code.md](../specs/phase3_training_code.md)
- **Status**: In Progress

---

## 1. Goal

Implement `src/training/` so that any of the 7 fine-tuned families (MobileNetV3-Small, VGG16, ResNet152, EfficientNet-B0, GoogLeNet, ViT-B/16, AlexNet) can be trained on either input path (full image or ROI) with one command. Training reads the frozen manifests, selects the checkpoint with the best validation macro-F1, resumes after interruption, and writes per-image prediction files for T08.

---

## 2. Requirements & Business Logic

1. **Manifest-Driven Data** (`data.py`):
   - Full path: resolve each row with `find_source_image()` from `src/data/create_roi_dataset.py`. ROI path: `image_root / roi_relative_path`.
   - Full path row filter follows spec decision D1 (`--full-scope matched|all`, default `matched` = the 1,100 rows also in `data/roi/roi_manifest.csv`).
   - Resolve every path at start-up and fail with the list of missing rows if any image is absent.
   - Run the case-leakage assertion before training.
   - Class list = sorted `canonical_wound_label` values; assert exactly 7 and that every class is present in each split.
   - In-memory cache of images resized to longest side 2 x input size (`--no-cache` to disable).
2. **Transforms** (`transforms.py`):
   - EXIF transpose → RGB → `PadToSquare` (D2) → resize → (train only) flip H/V, rotation ±15°, random resized crop scale 0.85–1.0, brightness/contrast 0.2, `AddGaussianNoise(σ=0.02, p=0.3)` → ImageNet normalization.
   - Eval transform has no randomness.
3. **Models** (`models.py`):
   - `build_model(family, num_classes=7, pretrained=True)` replaces the final classifier with a 7-class `Linear`; GoogLeNet without auxiliary heads.
   - `input_size(family)`: 384, except `vit_b16` = 224. Overridable with `--image-size` (tests only).
   - `feature_extractor(family)` returns the backbone with its head removed (for T06).
4. **Training Loop** (`engine.py`, `train.py`):
   - AdamW lr 3e-5, weight decay 1e-4, batch 32, cross-entropy with label smoothing 0.05 and class weights per D3, constant lr (D4).
   - Max 200 epochs; early stopping patience 25 on validation macro-F1, ties broken by lower validation loss.
   - bf16 autocast on CUDA.
   - Seed everything from `--seed` (default 42), including DataLoader worker seeds.
   - Log one JSON line per epoch to stdout and append to `training_history.csv` (EXP001 columns).
5. **Checkpoint & Resume**:
   - Save `last.pt` every epoch, holding model, optimizer, epoch, early-stopping state, RNG states and history. Save `best.pt` on improvement.
   - `--resume` continues from `last.pt`. If the run already finished, skip training and only (re)write missing prediction files.
6. **Outputs** (spec §4.4):
   - `run_config.json` with every setting, class list, git commit, library versions, hostname and `SLURM_JOB_ID`.
   - `validation_predictions.csv` and `test_predictions.csv` from `best.pt`. Test is predicted only after training has ended.
7. **Automated Testing** (`tests/test_training.py`), using synthetic images in `tmp_path` and no downloads:
   - Smoke: every family trains 2 epochs on CPU with `--no-pretrained --smoke` (64 px; ViT 224 px) and writes all output files.
   - Resume: stop after epoch 1, rerun with `--resume`, history continues at epoch 2 with no duplicated rows.
   - Missing image → clear error; leaked case across splits → `AssertionError`.
   - Prediction CSV: probabilities per row sum to 1 (±1e-4), columns match spec §4.4, one row per image in the split.
   - `PadToSquare` output is square and keeps the original pixels centred.

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/training/__init__.py` | Package marker |
| `src/training/config.py` | `RunConfig` dataclass, CLI parsing, `run_config.json` |
| `src/training/data.py` | Dataset, path resolution, class list, cache |
| `src/training/transforms.py` | `PadToSquare`, `AddGaussianNoise`, `build_transforms` |
| `src/training/models.py` | `build_model`, `input_size`, `feature_extractor` |
| `src/training/engine.py` | Train/predict loop, early stopping, checkpoints |
| `src/training/train.py` | CLI entry point |
| `tests/test_training.py` | Smoke, resume and integrity tests |
| `runbook.md` | Add "Prefetch pretrained weights" and local test commands |

---

## 4. CLI Interface

```bash
# Real run (HPC; SLURM wrapper comes in T09)
python src/training/train.py \
  --family efficientnet_b0 \
  --input-path full \
  --manifest data/splits/case_level_split_manifest.csv \
  --image-root /project/a260012_FORWCL/raw_data/wound_ml_7classes \
  --output-dir results/full/efficientnet_b0 \
  --resume

# Smoke test on synthetic or real data (CPU)
python src/training/train.py --family resnet152 --input-path roi \
  --manifest <manifest> --image-root <root> --output-dir <tmp> \
  --smoke --no-pretrained --device cpu
```

`--family` values: `mobilenet_v3_small, vgg16, resnet152, efficientnet_b0, googlenet, vit_b16, alexnet`.

---

## 5. Acceptance Check

1. `pytest tests/test_training.py` passes with 0 failures. If torch is not installed locally, run it inside the container on Raptor:
   ```bash
   srun -p short -c 2 --mem=4G -t 0:30:00 \
     apptainer exec /common/sif/pytorch-24.11-py3.sif python -m pytest -q tests/test_training.py
   ```
2. Existing suites still pass: `tests/test_validate_metadata.py`, `tests/test_split_leakage.py`, `tests/test_roi_preprocessing.py`.
3. A smoke run writes `run_config.json`, `training_history.csv`, `validation_predictions.csv`, `test_predictions.csv`, `best.pt`, `last.pt`.
4. No code path reads the test split before training ends (checked by review: `test` loader is built only in the final prediction step).
5. Tick T05 in `ROADMAP.md` §9.
