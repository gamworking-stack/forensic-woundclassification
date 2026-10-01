# Phase 3 Specification: Training Code

- **Phase**: 3 — Training Code
- **Related Tickets**: T05 (Shared Training Library), T06 (CNN-SVM), T07 (AlexNet Ensemble), T08 (Evaluation Module), T08b (Hyperparameter Tuning)
- **Status**: Approved
- **Owner**: Forensic Wound Classification Team (Antigravity & Preenapun Saokhieo)

---

## 1. Objectives

Build one in-repo training library that every one of the 16 scenarios (8 model families x 2 input paths) runs through, so that the only difference between a full-image and an ROI scenario is the input image.

1. **T05 — Shared Training Library (`src/training/`)**: manifest-driven dataset for both input paths, shared transforms, a model builder for the 7 fine-tuned families, one train/validate loop with early stopping, checkpoint and resume, and per-image prediction files.
2. **T06 — CNN-SVM (`src/training/cnn_svm.py`)**: frozen pretrained features into an SVM, tuned on validation, refit on train+validation.
3. **T07 — AlexNet Ensemble (`src/training/ensemble.py`)**: 3 AlexNet members trained by T05 with different seeds, combined by soft voting.
4. **T08 — Evaluation Module (`src/evaluation/`)**: all proposal metrics computed from the prediction files, never from a live model.
5. **T08b — Hyperparameter Tuning**: small shared search space per family on validation only.

This spec fixes the design of T05 in full and the interfaces that T06–T08b depend on. Each later ticket gets its own detail when it is written.

### 1.1 Relation to earlier runs

EXP001–EXP006 (full image and researcher ROI, run on Raptor in Aug–Sep 2026, e.g. job `221479`) were produced by scripts that are not in this repo. They are **exploratory references, not thesis results**. Their settings are the starting defaults below, and their test numbers are a sanity range for T11 (e.g. EfficientNet-B0 full-image test macro-F1 0.754, ResNet152 0.722). Their output columns (`training_history.csv`, `test_predictions.csv` with `prob_<class>` columns) are kept so old and new runs can be compared side by side.

---

## 2. Inputs

| Input | Full-image path | ROI path |
|---|---|---|
| Manifest | `data/splits/case_level_split_manifest.csv` | `data/roi/roi_manifest.csv` |
| Image root (HPC) | `/project/a260012_FORWCL/raw_data/wound_ml_7classes` | `/project/a260012_FORWCL/derived_data/roi_union_p008` |
| Path resolution | `find_source_image()` from `src/data/create_roi_dataset.py` (raw tree is by raw `wound_label`, some filenames lack an extension) | `roi_relative_path` column |
| Rows used | 1,100 matched to ROI (D1); all 1,106 as sensitivity run | 1,100 (779 / 164 / 157) |

- Labels: `canonical_wound_label`. Class order is the sorted canonical list from `data/splits/case_level_split_report.json` (`contusion, cut wound, graze wound, gunshot wound, laceration, penetration wound, scratch`), saved in every run config.
- Container: `/common/sif/pytorch-24.11-py3.sif` (torch 2.6, torchvision 0.20, CUDA 12.6), 1x H100 per job.
- Pretrained weights: torchvision ImageNet weights, cached under `TORCH_HOME=/project/a260012_FORWCL/torch_cache`. Download once on the login node, because compute nodes may have no internet.

---

## 3. Training Protocol (shared by all fine-tuned families)

| Setting | Value | Source |
|---|---|---|
| Families (T05) | MobileNetV3-Small, VGG16, ResNet152, EfficientNet-B0, GoogLeNet, ViT-B/16, AlexNet (ensemble member) | ROADMAP §3 |
| Weights | ImageNet pretrained, **all layers fine-tuned**, new 7-class head | proposal; frozen-backbone EXP runs scored far lower |
| Input size | 384 x 384 for CNNs, 224 x 224 for ViT-B/16 | phase 1 spec §6 |
| Preprocessing | EXIF transpose → RGB → pad to square (D2) → resize → ImageNet normalization | proposal + D2 |
| Augmentation (train only) | horizontal + vertical flip; rotation ±15°; zoom via random resized crop, scale 0.85–1.0; brightness/contrast jitter 0.2; Gaussian noise σ = 0.02 with p = 0.3. **No hue/saturation jitter** (colour carries contusion information) | proposal list |
| Loss | cross-entropy, label smoothing 0.05, class weights per D3 | phase 1 spec §6 |
| Optimizer | AdamW, lr 3e-5, weight decay 1e-4, constant learning rate | phase 1 spec §6, EXP001 |
| Batch size | 32 | EXP001 |
| Epochs | max 200, early stopping patience 25 on **validation macro-F1** | phase 1 spec §6 |
| Precision | bf16 autocast on CUDA; fp32 on CPU | H100 |
| Seed | 42 (Python, NumPy, torch, DataLoader workers); ensemble members 43, 44, 45 | ROADMAP ground rules |

**Model selection:** the checkpoint with the highest validation macro-F1 is kept, with ties broken by lower validation loss. The test split is predicted **once**, after training ends, with that checkpoint.

---

## 4. T05 Design (`src/training/`)

```
src/training/
├── __init__.py
├── config.py       # RunConfig dataclass; CLI parsing; saved as run_config.json
├── data.py         # ManifestImageDataset, path resolution, class list, image cache
├── transforms.py   # PadToSquare, AddGaussianNoise, build_transforms(train|eval, size)
├── models.py       # build_model(family, num_classes, pretrained) and input size per family
├── engine.py       # train_one_epoch, predict, EarlyStopping, save/load checkpoint
└── train.py        # CLI entry point: one family x one input path per call
```

### 4.1 Data
- `ManifestImageDataset(manifest, image_root, input_path, split, classes, transform)`.
- Every image path is resolved once at start-up. A missing file is a hard error that lists the missing rows; nothing is skipped silently.
- The case-leakage assertion from `src/data/create_case_split.verify_leakage` runs on the manifest before training.
- **Image cache**: the raw photos are up to 6000 x 4000, so each image is decoded once and kept in memory with its longest side resized to 2 x input size, before any augmentation (about 2 GB at 384 px). Disable with `--no-cache`.

### 4.2 Models
- torchvision builders: `mobilenet_v3_small`, `vgg16`, `resnet152`, `efficientnet_b0`, `googlenet` (no auxiliary heads), `vit_b_16`, `alexnet`. The last classifier layer is replaced by `Linear(in_features, 7)`.
- `build_model(..., pretrained=False)` must work offline for tests.
- `feature_extractor(family)` returns the backbone without its head; T06 uses this.

### 4.3 Engine
- `train_one_epoch` and `predict(model, loader) -> (probabilities, labels, loss)` are shared by training, validation and test.
- Checkpoints: `last.pt` after every epoch (model, optimizer, epoch, early-stopping state, RNG states, history) and `best.pt` on improvement.
- `--resume` continues from `last.pt` at the next epoch with identical early-stopping state. A run that finished early stopping is detected and skipped.

### 4.4 Outputs (one folder per scenario)
```
<output-dir>/                       # default results/<input_path>/<family>/
├── run_config.json                 # every setting, class list, git commit, torch/CUDA versions, host, SLURM job ID
├── training_history.csv            # same columns as EXP001
├── validation_predictions.csv      # best checkpoint
├── test_predictions.csv            # best checkpoint, written once
├── best.pt, last.pt                # gitignored; stay on the HPC
└── train.log
```
Prediction CSV columns: `relative_path, pseudonimized_id, split, true_class, predicted_class, prob_<class>...`, with class names written as `prob_cut_wound`, etc. T08 computes every metric from these files.

### 4.5 CLI
```bash
python src/training/train.py \
  --family resnet152 \
  --input-path roi \
  --manifest data/roi/roi_manifest.csv \
  --image-root /project/a260012_FORWCL/derived_data/roi_union_p008 \
  --output-dir results/roi/resnet152 \
  --resume
```
Flags for tests: `--smoke` (2 epochs on 2 images per class per split), `--no-pretrained`, `--device cpu`.

---

## 5. Interfaces for Later Tickets

- **T06 CNN-SVM** uses `data.py`, `transforms.py` (eval transform only) and `models.feature_extractor`. It writes the same prediction CSVs, with probabilities from Platt scaling (`SVC(probability=True)`).
- **T07 Ensemble** calls `train.py` 3 times with `--family alexnet --seed 43|44|45`, then averages the members' `prob_*` columns into the ensemble's prediction CSVs.
- **T08 Evaluation** reads only prediction CSVs and produces accuracy, balanced accuracy, macro/class-wise precision, recall and F1, confusion matrix, and one-vs-rest ROC/AUC.
- **T08b Tuning** passes overrides (`--lr`, `--batch-size`, `--dropout`) to `train.py`, and selects on validation macro-F1 only.

### 5.1 Scenario IDs

| Family | Full | ROI |
|---|---|---|
| EfficientNet-B0 | S01 | S02 |
| ResNet152 | S03 | S04 |
| MobileNetV3-Small | S05 | S06 |
| GoogLeNet | S07 | S08 |
| VGG16 | S09 | S10 |
| ViT-B/16 | S11 | S12 |
| CNN-SVM | S13 | S14 |
| AlexNet ensemble | S15 | S16 |

---

## 6. Decisions (confirmed 2026-10-01)

| ID | Question | Decision | Why |
|---|---|---|---|
| D1 | Which images does the full-image path use? | **Matched: the same 1,100 images as ROI** (drop the 6 unannotated), so each full/ROI pair differs only in the input. Report the all-1,106 full run as a sensitivity check. | Cleanest answer to "does ROI improve over full image?" EXP001 used all 1,106. |
| D2 | Pad to square before resize? | **Yes**, constant black padding | Keeps wound shape; stretching distorts thin scratches and cuts |
| D3 | Class-weighted loss? | **Yes**, inverse class frequency on the train split, normalized to mean 1 | Penetration (52 images) vs gunshot (310); macro metrics are primary |
| D4 | Learning-rate schedule | **Constant** 3e-5, as in EXP001 | Keeps the comparison with earlier runs simple; T08b may revisit |

---

## 7. Acceptance Checks (Phase 3)

1. **T05**: synthetic-data smoke test (2 epochs, `--no-pretrained`, CPU) passes for every family; resume test passes; leakage and missing-file errors are raised. Details in `tickets/T05_shared_training_library.md`.
2. **T06–T08b**: checks set in their own tickets.
3. **Phase gate**: T11 (2-epoch HPC smoke on real images) runs one family on both input paths through the same code with no edits.
