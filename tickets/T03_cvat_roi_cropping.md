# Ticket T03: CVAT YOLO Annotation Parsing & Padded Union Bounding Box Cropping

- **Ticket ID**: `T03`
- **Phase**: Phase 2 (ROI Preprocessing)
- **Spec Reference**: [specs/phase2_roi_preprocessing.md](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/specs/phase2_roi_preprocessing.md)
- **Status**: Completed

---

## 1. Goal

Implement `src/data/create_roi_dataset.py` to parse CVAT YOLO 1.1 annotation zip exports, match them against the frozen case-level split manifest (`data/splits/case_level_split_manifest.csv`), calculate padded union bounding boxes (8% padding, min 8px dimension), crop the visible wound regions, and output `data/roi/roi_manifest.csv` and `data/roi/crop_summary.json`.

---

## 2. Requirements & Business Logic

1. **CVAT YOLO 1.1 Ingestion**:
   - Ingest multiple annotation exports (`.zip` files).
   - Read class names from `obj.names` and normalized boxes (`class_id, x_center, y_center, w, h`) from `obj_train_data/*.txt`.
   - Normalize label names to canonical wound classes using alias mapping.
2. **Padded Union Bounding Box**:
   - For images with multiple boxes, merge them into a single bounding box enclosing all visible wounds ($x_1 = \min(x_{1,i}), y_1 = \min(y_{1,i}), x_2 = \max(x_{2,i}), y_2 = \max(y_{2,i})$).
   - Apply symmetric padding of 8% ($p = 0.08$):
     $\Delta X = (x_2 - x_1) \times 0.08$, $\Delta Y = (y_2 - y_1) \times 0.08$.
   - Clamp coordinates to image boundaries $[0, W]$ and $[0, H]$.
   - Skip crops smaller than minimum threshold (8 px).
3. **Split and Case Inheritance**:
   - Every ROI crop strictly inherits `pseudonimized_id` and `split` (`train`, `validation`, `test`) from its source image in `case_level_split_manifest.csv`.
   - No case leakage allowed across splits.
4. **Execution Modes**:
   - Live mode: loads image with `ImageOps.exif_transpose`, crops box, saves JPEG at quality 95 to `output_dir/<split>/<canonical_class>/<stem>_roi.jpg`.
   - Dry-run mode (`--dry-run`): parses annotations, validates coordinates, and generates `roi_manifest.csv` and `crop_summary.json` without requiring image files on disk.
5. **Artifacts to Generate**:
   - `data/roi/roi_manifest.csv` (1,100 records with crop box coordinates).
   - `data/roi/crop_summary.json` (summary statistics).

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| [`src/data/create_roi_dataset.py`](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/src/data/create_roi_dataset.py) | CLI utility for CVAT parsing, union box calculation, and cropping |
| `data/roi/roi_manifest.csv` | Output manifest for all ROI cropped images |
| `data/roi/crop_summary.json` | JSON summary of crop operations |

---

## 4. CLI Interface

```bash
python3 src/data/create_roi_dataset.py \
  --manifest "data/splits/case_level_split_manifest.csv" \
  --exports /Users/preenapunsaokhieo/Downloads/task_*_annotations_2026_09_30_*.zip \
  --output-dir "data/roi" \
  --padding 0.08 \
  --min-size 8 \
  --crop-mode union
```

---

## 5. Acceptance Check

1. Script runs and exits with code 0.
2. Resulting manifest `data/roi/roi_manifest.csv` contains exactly **1,100 rows** across **420 cases**.
   - *Implementation note*: the observed count is **418 cases**, not 420. Three of the 6 unannotated images are the only image of their case (220 `PC235480.JPG`, 353 `P1070307.JPG`, 415 `P1071425`), so 421 − 3 = 418. The earlier HPC reference run (`Researcher_ROI_results`) also has 418. The 420 in this ticket, the Phase 2 spec and T04 needs confirming.
3. Zero leakage assertion passes: train, val, and test case sets are mutually disjoint.
4. All 7 canonical classes are present in `train` (779), `validation` (164), and `test` (157).
5. `data/roi/crop_summary.json` lists 6 unannotated records, matching the known dataset profile.
