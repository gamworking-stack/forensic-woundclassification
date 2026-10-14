# Phase 2 Specification: ROI Preprocessing and Full vs. ROI Audit

- **Phase**: 2 — ROI Preprocessing
- **Related Tickets**: T03 (CVAT Annotation Parsing & Bounding Box Cropping), T04 (Full vs. ROI Alignment Audit)
- **Status**: Completed (outcome in §6; the 420-case target in §1 and §5 is corrected there to 418)
- **Owner**: Forensic Wound Classification Team (Antigravity & Preenapun Saokhieo)

---

## 1. Objectives

1. **T03 — CVAT Annotation Parsing & Bounding Box Cropping (`src/data/create_roi_dataset.py`)**:
   - Ingest CVAT YOLO 1.1 annotation archives (`.zip` exports).
   - Align annotations with the frozen case-level split manifest (`data/splits/case_level_split_manifest.csv`).
   - For images containing multiple annotations, calculate the **union bounding box** so that each source image maps to exactly one ROI sample.
   - Apply standard symmetric padding (default **8%**, $0.08$) and assert a minimum crop dimension (default **8 px**).
   - Guarantee strict **case-level split inheritance**: each cropped image inherits the exact `pseudonimized_id` and `split` (`train`, `validation`, `test`) from its source image.
   - Provide a `--dry-run` flag to parse, validate coordinates, compute union boxes, and generate the `roi_manifest.csv` even when raw source image binaries reside remotely on the MedCMU-HPC cluster.

2. **T04 — Full vs. ROI Dataset Alignment Audit (`src/data/audit_full_vs_roi.py`)**:
   - Formally compare the full-image dataset (1,106 images across 421 cases) with the ROI dataset (1,100 images across 420 cases).
   - Identify and document the **6 unannotated / missing images** to guarantee fair scientific comparison for the thesis core research question: *Does ROI localization improve classification performance over full forensic images?*
   - Output comparative distribution tables and JSON audit reports documenting case and image counts per class and split.

---

## 2. Inputs & Source Material

### 2.1 Inputs
- **Authoritative Split Manifest**: [`data/splits/case_level_split_manifest.csv`](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/data/splits/case_level_split_manifest.csv) (1,106 rows, 421 unique cases).
- **CVAT Annotation Exports**: 7 YOLO 1.1 `.zip` archives containing `obj.names` and normalized bounding boxes (`obj_train_data/*.txt`):
  - `task_12_annotations_2026_09_30_16_02_03_yolo 1.1.zip`
  - `task_14_annotations_2026_09_30_16_00_10_yolo 1.1.zip`
  - `task_15_annotations_2026_09_30_15_39_29_yolo 1.1.zip`
  - `task_16_annotations_2026_09_30_15_34_41_yolo 1.1.zip`
  - `task_17_annotations_2026_09_30_15_31_44_yolo 1.1.zip`
  - `task_18_annotations_2026_09_30_15_17_53_yolo 1.1.zip`
  - `task_19_annotations_2026_09_30_15_14_48_yolo 1.1.zip`
- **Raw Image Root**:
  - Local external storage: `/Volumes/GAMMY_DRIVE/wound ml`
  - HPC cluster: `/project/a260012_FORWCL/raw_data/wound_ml_7classes`

---

## 3. Algorithm & Implementation Details

### 3.1 T03: ROI Cropping Logic (`src/data/create_roi_dataset.py`)

#### 1. Annotation Parsing & Normalization
YOLO format expresses boxes as normalized floats:
$$\left[ \text{class\_id}, x_{\text{center}}, y_{\text{center}}, w, h \right]$$
Convert to absolute pixel coordinates on image dimensions $(W, H)$:
$$x_1 = (x_{\text{center}} - w / 2) \cdot W, \quad y_1 = (y_{\text{center}} - h / 2) \cdot H$$
$$x_2 = (x_{\text{center}} + w / 2) \cdot W, \quad y_2 = (y_{\text{center}} + h / 2) \cdot H$$

#### 2. Union Bounding Box
When an image contains $K \ge 1$ boxes:
$$X_1 = \min_{k} (x_{1, k}), \quad Y_1 = \min_{k} (y_{1, k})$$
$$X_2 = \max_{k} (x_{2, k}), \quad Y_2 = \max_{k} (y_{2, k})$$

#### 3. Padding & Clamping
With padding ratio $p = 0.08$ (8%):
$$\Delta X = (X_2 - X_1) \cdot p, \quad \Delta Y = (Y_2 - Y_1) \cdot p$$
$$X_1^{\text{pad}} = \max(0, \text{round}(X_1 - \Delta X)), \quad Y_1^{\text{pad}} = \max(0, \text{round}(Y_1 - \Delta Y))$$
$$X_2^{\text{pad}} = \min(W, \text{round}(X_2 + \Delta X)), \quad Y_2^{\text{pad}} = \min(H, \text{round}(Y_2 + \Delta Y))$$

#### 4. Minimum Dimension Filter
Require $(X_2^{\text{pad}} - X_1^{\text{pad}}) \ge 8$ and $(Y_2^{\text{pad}} - Y_1^{\text{pad}}) \ge 8$.

#### 5. Image Transformation & Output Path
- Use `PIL.ImageOps.exif_transpose` to respect EXIF camera orientation.
- Save as JPEG at quality 95:
  `output_dir/<split>/<canonical_wound_label>/<filename_stem>_roi.jpg`
- Emit `data/roi/roi_manifest.csv` and `data/roi/crop_summary.json`.

---

### 3.2 T04: Full vs. ROI Alignment Audit (`src/data/audit_full_vs_roi.py`)

1. Load both manifests:
   - Full image: `data/splits/case_level_split_manifest.csv` (1,106 rows)
   - ROI crops: `data/roi/roi_manifest.csv` (1,100 rows)
2. Identify records missing from ROI:
   - Identify the 6 unannotated records.
   - Characterize missing instances by class, split, and case ID:
     - Case 94: `ZZ140321.JPG` (scratch, validation)
     - Case 167: `ZZ310855.JPG` (contusion, train)
     - Case 220: `PC235480.JPG` (scratch, validation)
     - Case 353: `P1070307.JPG` (scratch, test)
     - Case 388: `IMG_3520.JPG` (scratch, train)
     - Case 415: `P1071425` (cut wound, train)
3. Generate comparative breakdown:
   - Image count comparison per class $\times$ split (Full vs ROI).
   - Case count comparison per class $\times$ split (Full vs ROI).
   - Retention rates (e.g. Gunshot: 100%, Graze: 100%, Laceration: 100%, Contusion: 99.5%, Cut: 98.9%, Penetration: 100%, Scratch: 94.2%).
4. Save:
   - `data/audits/full_vs_roi_audit_table.csv`
   - `data/audits/full_vs_roi_audit_report.json`

---

## 4. Expected Deliverables & File Layout

```
data/
├── roi/
│   ├── roi_manifest.csv               # 1,100 ROI records with crop coordinates
│   └── crop_summary.json              # Processing statistics & counts
└── audits/
    ├── full_vs_roi_audit_table.csv    # Comparative metrics table
    └── full_vs_roi_audit_report.json  # Analysis report on unannotated samples
src/
└── data/
    ├── create_roi_dataset.py          # T03 implementation
    └── audit_full_vs_roi.py           # T04 implementation
tests/
└── test_roi_preprocessing.py          # Unit & regression tests for ROI pipeline
```

---

## 5. Acceptance Checks

1. **T03 (ROI Cropping)**:
   - CLI script supports `--dry-run` and live execution with exit code 0.
   - Correctly maps all 7 CVAT export zips without annotation collisions.
   - Resulting ROI manifest contains exactly **1,100 records** across **420 cases**.
   - Zero case leakage: $\text{train\_cases} \cap \text{val\_cases} = \emptyset$, $\text{train\_cases} \cap \text{test\_cases} = \emptyset$, $\text{val\_cases} \cap \text{test\_cases} = \emptyset$.
   - All 7 canonical classes represented in Train (779), Validation (164), and Test (157).
2. **T04 (Full vs. ROI Audit)**:
   - CLI audit tool runs cleanly and produces comparative reports.
   - Exactly 6 unannotated images documented with zero discrepancies.
   - Verifies that Test set retains all 7 classes in both Full (158 images) and ROI (157 images, 1 scratch dropped).
3. **Unit Tests**:
   - `python3 tests/test_roi_preprocessing.py` passes with exit code 0.

---

## 6. Outcome (closed 2026-10-14)

| Check | Result |
|---|---|
| ROI crops (T03) | dry run and live run exit 0 (SLURM job `221586`); 7 exports, 2 browser duplicates skipped; **1,100 records from 418 cases**; 779 / 164 / 157 images; every class in every split; no case leakage |
| Audit (T04) | 6 unannotated images documented, 0 discrepancies; test keeps all 7 classes (158 full, 157 ROI, 1 scratch dropped) |
| Unit tests | `tests/test_roi_preprocessing.py` passes |

**Correction to §1 and §5**: the ROI set has 418 cases, not 420. Three of the 6 unannotated images are the only image of their case (cases 220, 353, 415), so 421 − 3 = 418 (`data/audits/full_vs_roi_audit_report.json`, `cases_lost: 3`). The 420 was an arithmetic slip in this spec; no data changed.

Details: `tickets/T03_cvat_roi_cropping.md`, `tickets/T04_full_vs_roi_audit.md`.
