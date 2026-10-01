# Ticket T01: Metadata Validation and Label Canonicalization

- **Ticket ID**: `T01`
- **Phase**: Phase 1 (Dataset and Split)
- **Spec Reference**: [specs/phase1_dataset_and_split.md](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/specs/phase1_dataset_and_split.md)
- **Status**: Completed

---

## 1. Goal

Implement `src/data/validate_metadata.py` to ingest the project's metadata (`thesis_data_30_sep_id_only.csv` or `thesis_data_30_sep.xlsx`), validate data integrity against the thesis requirements, normalize labels to the 7 canonical classes, and generate a machine-readable validation report with dynamic counts.

---

## 2. Requirements & Business Logic

1. **Schema Validation**:
   - Required columns: `pseudonimized_id`, `filename`, `wound_label` (and preserve `wound_id` if present).
   - Check for missing/null/whitespace values in required columns.
2. **Label Canonicalization**:
   - Normalize raw string labels to the 7 formal classes:
     - `cut` $\rightarrow$ `cut wound`
     - `graze` $\rightarrow$ `graze wound`
     - `gsw` $\rightarrow$ `gunshot wound`
     - `contusion` $\rightarrow$ `contusion`
     - `laceration` $\rightarrow$ `laceration`
     - `scratch` $\rightarrow$ `scratch`
     - `penetrating wound (sharp&blunt)` $\rightarrow$ `penetration wound`
   - Raise a descriptive error if an unknown label is encountered.
3. **Filename Integrity & Duplication Checks**:
   - Flag any duplicate filenames mapped to conflicting cases or labels.
   - Detect filename anomalies (e.g. missing extension `P1280417` vs `P1280417.JPG`) and standardize relative path generation.
4. **Dynamic Reporting (No Hard-coded Counts)**:
   - Output structured JSON report recording:
     - Total image count.
     - Total unique case count (`pseudonimized_id`).
     - Image count per canonical class.
     - Case count per canonical class.
     - Multi-wound case frequency (cases with $>1$ image or $>1$ wound type).
     - Any validation warnings or errors.

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| [`src/data/validate_metadata.py`](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/src/data/validate_metadata.py) | CLI utility to validate metadata and output clean dataset CSV + report |
| [`src/data/__init__.py`](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/src/data/__init__.py) | Module initializer |
| `data/validated_metadata.csv` | Output cleaned and canonicalized metadata |
| `data/metadata_validation_report.json` | Generated validation report |

---

## 4. CLI Interface

```bash
python src/data/validate_metadata.py \
  --input-csv "../thesis_data_30_sep_id_only.csv" \
  --output-csv "data/validated_metadata.csv" \
  --output-report "data/metadata_validation_report.json"
```

---

## 5. Acceptance Check

Run the CLI command and verify:
1. Command exits with code 0.
2. `data/metadata_validation_report.json` contains `"errors": []` and `"status": "passed"`.
3. All 7 canonical classes are detected.
4. Total images and cases are computed dynamically and match the dataset (1,106 images, 421 cases).
5. `data/validated_metadata.csv` contains all records with canonical labels.
