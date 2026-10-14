# Phase 1 Specification: Dataset Validation and Case-Level Split

- **Phase**: 1 — Dataset and Split
- **Related Tickets**: T01 (Metadata validation & label canonicalization), T02 (Case-level 70/15/15 grouped split)
- **Status**: Completed (outcome in §7)
- **Owner**: Forensic Wound Classification Team (Antigravity & Preenapun Saokhieo)

---

## 1. Objectives

1. **T01 — Metadata Validation & Canonicalization**:
   - Ingest and validate the metadata source (`thesis_data_30_sep_id_only.csv` or `thesis_data_30_sep.xlsx`).
   - Check required schema: `pseudonimized_id`, `filename`, `wound_label`, optional `wound_id`.
   - Ensure zero null values, no duplicate file mappings, and resolve case-insensitive file extensions (e.g., bare IDs vs `.JPG`/`.jpg`).
   - Canonicalize labels to the 7 formal thesis classes.
   - Generate validation report with dynamic case/image counts (strictly zero hard-coded values).

2. **T02 — Case-Level Leakage-Free 70/15/15 Split**:
   - Partition cases (`pseudonimized_id`) into **70% Train / 15% Validation / 15% Test**.
   - Ensure **zero case leakage**: all images belonging to the same `pseudonimized_id` must reside in one and only one split partition.
   - Employ a seeded multi-objective search (seed 42) to optimize class balance across partitions despite cases having multi-wound / multi-label occurrences.
   - Guarantee that all 7 classes exist in all three subsets (`train`, `validation`, `test`).
   - Output, freeze, and commit the official split manifest to `data/splits/case_level_split_manifest.csv`.

---

## 2. Inputs & Canonical Mapping

### 2.1 Metadata Sources
- Primary ID CSV: `../thesis_data_30_sep_id_only.csv` (1,106 rows, 421 unique cases).
- Extended Metadata Excel: `../thesis_data_30_sep.xlsx` (includes clinical attributes: `age`, `autopsy`, `body_site`, `race_ethnicity`, `skin_tone`).
- Raw Images Root:
  - Local: `/Volumes/GAMMY_DRIVE/wound ml` (when mounted)
  - MedCMU-HPC Raptor: `/project/a260012_FORWCL/raw_data/wound_ml_7classes`

### 2.2 Canonical 7-Class Label Mapping

| Raw CSV Label (`wound_label`) | Canonical Label (`canonical_wound_label`) | Expected Class Index |
|---|---|---|
| `cut` | `cut wound` | 0 |
| `graze` | `graze wound` | 1 |
| `gsw` | `gunshot wound` | 2 |
| `contusion` | `contusion` | 3 |
| `laceration` | `laceration` | 4 |
| `scratch` | `scratch` | 5 |
| `penetrating wound (sharp&blunt)` | `penetration wound` | 6 |

---

## 3. Detailed Component Specification

### 3.1 T01: Validation Module (`src/data/validate_metadata.py`)
- **Inputs**:
  - `--input-csv` or `--input-excel`: Path to source metadata.
  - `--image-root`: (Optional) Root directory to check physical file existence.
  - `--output-report`: Path to JSON validation summary.
- **Rules & Checks**:
  1. Required columns exist: `pseudonimized_id`, `filename`, `wound_label`.
  2. No null/NaN/whitespace-only values in required fields.
  3. Every `wound_label` maps cleanly to one of the 7 canonical classes.
  4. Filename format normalization: automatically check with/without `.JPG`, `.jpg`, `.png`.
  5. Check for duplicate image entries within the same case or across cases.
  6. Output validation summary JSON containing:
     - `total_images`: total valid rows
     - `total_cases`: number of unique `pseudonimized_id`
     - `per_class_image_counts`: dictionary of counts
     - `per_class_case_counts`: dictionary of case occurrences
     - `errors`: list of validation errors (must be empty to proceed)

### 3.2 T02: Grouped Split Generator (`src/data/create_case_split.py`)
- **Inputs**:
  - `--input-csv`: Validated metadata CSV.
  - `--output-dir`: Target output directory (`data/splits/`).
  - `--seed`: Default `42`.
  - `--trials`: Default `50,000` random trials.
  - `--ratios`: `0.70, 0.15, 0.15`.
- **Optimization Strategy**:
  - Target group counts: $N_{train} = \text{round}(0.70 \times N)$, $N_{val} = \text{round}(0.15 \times N)$, $N_{test} = N - N_{train} - N_{val}$.
  - For each trial, randomly assign group IDs according to target partition sizes.
  - Compute deviation score: combination of group size deviations and class-level image/case proportion deviations across train, val, and test.
  - Filter constraint: every class must have at least $\min(C_{total}, 1)$ instances in train, validation, and test.
  - Select candidate with minimum loss.
- **Strict Leakage Assertion**:
  ```python
  train_cases = set(df[df['split'] == 'train']['pseudonimized_id'])
  val_cases = set(df[df['split'] == 'validation']['pseudonimized_id'])
  test_cases = set(df[df['split'] == 'test']['pseudonimized_id'])
  assert train_cases.isdisjoint(val_cases), "Data leakage between train and validation!"
  assert train_cases.isdisjoint(test_cases), "Data leakage between train and test!"
  assert val_cases.isdisjoint(test_cases), "Data leakage between validation and test!"
  ```

---

## 4. Expected Deliverables & File Layout

```
data/
└── splits/
    ├── case_level_split_manifest.csv   # Frozen manifest with 'split' column
    ├── case_level_split_summary.csv    # Tabulated counts per class and split
    └── case_level_split_report.json    # Run parameters, seed, leakage status
src/
└── data/
    ├── __init__.py
    ├── validate_metadata.py            # T01 implementation
    └── create_case_split.py            # T02 implementation
tests/
└── test_split_leakage.py               # Automated pytest verifying zero leakage
```

---

## 5. Acceptance Checks

1. **Validation Report (T01)**:
   - Zero validation errors reported on the dataset.
   - All 7 wound classes verified present and correctly mapped.
   - Exactly 1,106 image records and 421 cases identified (or dynamic equivalent from input).

2. **Split Manifest (T02)**:
   - Output manifest `data/splits/case_level_split_manifest.csv` is generated and committed.
   - Target split distribution achieved:
     - Group counts: 295 Train (~70%), 63 Validation (~15%), 63 Test (~15%) across 421 cases.
     - Class presence: All 7 classes present in train, validation, and test.
   - `test_split_leakage.py` passes with exit code 0.
   - Once generated, manifest is marked **FROZEN** and will be consumed by Phase 2 (ROI cropping) and Phase 3 (Model training).

---

## 6. Alignment on Roadmap Section 8 Inquiries

Based on project configuration on the MedCMU-HPC cluster:
1. **Cluster Configuration**:
   - Host/Portal: `raptor.med.cmu.ac.th:9000` / SSH `raptor`
   - Account: `a260012`
   - Partition: `gpu` (`--gres=gpu:1`, 12 CPUs, 96GB RAM)
   - Cluster Project Root: `/project/a260012_FORWCL`
   - Container Image: `/common/sif/pytorch-24.11-py3.sif` (`apptainer/1.3.5`)
2. **Training Budget**:
   - Max 200 epochs, early stopping patience 25 on validation macro-F1, AdamW (lr 3e-5, weight decay 1e-4), label smoothing 0.05.
3. **Special Architectures**:
   - AlexNet Ensemble: 3 models trained with distinct seeds, soft-voting inference.
   - Hybrid CNN-SVM: feature extraction from VGG16 + ResNet152 + EfficientNet-B0 into LinearSVC/RBF SVM.
   - Grad-CAM: evaluated for 6 direct CNNs + ViT-B/16 (with token reshape); 1 representative member for AlexNet ensemble.
4. **Data Privacy**:
   - Patient imagery remains on secure HPC storage (`/project/a260012_FORWCL/raw_data`) and local encrypted volume.
   - Only code, manifests, and aggregated metrics/figures are tracked in Git.

---

## 7. Outcome (closed 2026-10-14)

| Check | Result |
|---|---|
| Validation report (T01) | 0 errors; 7 classes; 1,106 records, 421 cases (`data/metadata_validation_report.json`); 2 warnings kept: 61 filenames without extension, 3 shared filenames |
| Split (T02) | 295 / 63 / 63 cases, 782 / 166 / 158 images, every class in every split, seed 42 over 50,000 trials (`data/splits/case_level_split_report.json`, `frozen: true`) |
| Leakage | `tests/test_split_leakage.py` passes; re-checked on every run in T16 (test and validation cases match the frozen split) |
| Frozen | the manifest has not changed since it was committed; Phases 2 to 6 read it unchanged |

Details: `tickets/T01_validate_metadata.md`, `tickets/T02_case_level_split.md`.
