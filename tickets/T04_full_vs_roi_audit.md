# Ticket T04: Full Image vs. ROI Alignment Audit

- **Ticket ID**: `T04`
- **Phase**: Phase 2 (ROI Preprocessing)
- **Spec Reference**: [specs/phase2_roi_preprocessing.md](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/specs/phase2_roi_preprocessing.md)
- **Status**: Open / Blocked on T03

---

## 1. Goal

Implement `src/data/audit_full_vs_roi.py` and `tests/test_roi_preprocessing.py` to systematically audit the full-image dataset (1,106 records, 421 cases) against the ROI dataset (1,100 records, 420 cases). Document the 6 unannotated images, compute class and split coverage, and ensure fair comparability for the thesis research question.

---

## 2. Requirements & Business Logic

1. **Cross-Manifest Audit**:
   - Compare `data/splits/case_level_split_manifest.csv` and `data/roi/roi_manifest.csv`.
   - Identify unannotated images that could not be cropped.
   - Specifically log the 6 known unannotated records:
     - Case 94: `ZZ140321.JPG` (`scratch`, validation)
     - Case 167: `ZZ310855.JPG` (`contusion`, train)
     - Case 220: `PC235480.JPG` (`scratch`, validation)
     - Case 353: `P1070307.JPG` (`scratch`, test)
     - Case 388: `IMG_3520.JPG` (`scratch`, train)
     - Case 415: `P1071425` (`cut wound`, train)
2. **Distribution & Retention Comparison**:
   - Generate a side-by-side comparative table showing Full Image vs. ROI for every canonical class across `train`, `validation`, and `test`.
   - Compute retention rates (images and cases).
   - Confirm that the test set retains all 7 classes in both paths (Full: 158 images; ROI: 157 images, exactly 1 test scratch unannotated).
3. **Artifacts to Generate**:
   - `data/audits/full_vs_roi_audit_table.csv`
   - `data/audits/full_vs_roi_audit_report.json`
4. **Automated Testing**:
   - Build `tests/test_roi_preprocessing.py` testing union box math, padding, split inheritance, and audit consistency.

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| [`src/data/audit_full_vs_roi.py`](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/src/data/audit_full_vs_roi.py) | CLI utility to perform cross-manifest audit |
| `data/audits/full_vs_roi_audit_table.csv` | Comparative counts table per split and class |
| `data/audits/full_vs_roi_audit_report.json` | Detailed JSON audit report |
| [`tests/test_roi_preprocessing.py`](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/tests/test_roi_preprocessing.py) | Test suite for ROI bounding box math and audit logic |

---

## 4. CLI Interface

```bash
python3 src/data/audit_full_vs_roi.py \
  --full-manifest "data/splits/case_level_split_manifest.csv" \
  --roi-manifest "data/roi/roi_manifest.csv" \
  --output-dir "data/audits"
```

---

## 5. Acceptance Check

1. Script executes and exits with code 0.
2. Identifies exactly the 6 unannotated records.
3. Produces `data/audits/full_vs_roi_audit_table.csv` and `data/audits/full_vs_roi_audit_report.json`.
4. Run `python3 tests/test_roi_preprocessing.py`:
   - All tests pass with exit code 0.
   - Confirms case disjointness and test class completeness for both Full and ROI paths.
