# Ticket T02: Case-Level 70/15/15 Grouped Split

- **Ticket ID**: `T02`
- **Phase**: Phase 1 (Dataset and Split)
- **Spec Reference**: [specs/phase1_dataset_and_split.md](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/specs/phase1_dataset_and_split.md)
- **Status**: Open / Blocked on T01

---

## 1. Goal

Implement `src/data/create_case_split.py` to partition the dataset into a **70% Train / 15% Validation / 15% Test** split grouped strictly by `pseudonimized_id` (zero case leakage). Use a seeded multi-objective search (seed `42`, 50,000 trials) to balance class distribution across partitions and guarantee all 7 classes are present in every partition. Freeze the split manifest as the single source of truth for all 16 experimental scenarios.

---

## 2. Requirements & Business Logic

1. **Strict Zero-Leakage Guarantee**:
   - Every `pseudonimized_id` must belong to exactly one split partition: `train`, `validation`, or `test`.
   - $\text{train\_cases} \cap \text{val\_cases} = \emptyset$
   - $\text{train\_cases} \cap \text{test\_cases} = \emptyset$
   - $\text{val\_cases} \cap \text{test\_cases} = \emptyset$
2. **Multi-Objective Stratified Balancing**:
   - Group ratio targets: 70% Train, 15% Validation, 15% Test (approx. 295 Train, 63 Validation, 63 Test groups).
   - Because cases can have multiple wounds and multiple labels, search across candidate partitions to minimize the discrepancy between partition class proportions and global class proportions.
   - Hard constraint: All 7 classes must appear in `train`, `validation`, and `test` partitions.
3. **Artifact Freezing & Reproducibility**:
   - Fixed random seed: `42`.
   - Output manifest committed to Git: `data/splits/case_level_split_manifest.csv`.
   - Never regenerate this split once committed.
4. **Automated Verification**:
   - Create `tests/test_split_leakage.py` using `pytest`.

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| [`src/data/create_case_split.py`](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/src/data/create_case_split.py) | Script to generate the optimal grouped split |
| `data/splits/case_level_split_manifest.csv` | The single frozen split manifest |
| `data/splits/case_level_split_summary.csv` | Summary table of image & case counts per class per split |
| `data/splits/case_level_split_report.json` | Reproducibility report (seed, score, distribution stats) |
| [`tests/test_split_leakage.py`](file:///Users/preenapunsaokhieo/Desktop/%5BThesis%5D%20Wound%20classification/forensic-woundclassification/tests/test_split_leakage.py) | Unit tests verifying zero case leakage and split integrity |

---

## 4. CLI Interface

```bash
python src/data/create_case_split.py \
  --input-csv "data/validated_metadata.csv" \
  --output-dir "data/splits" \
  --seed 42 \
  --trials 50000
```

---

## 5. Acceptance Check

1. Run the split generation script:
   - Exits with code 0.
   - Outputs `case_level_split_manifest.csv`, `case_level_split_summary.csv`, and `case_level_split_report.json`.
2. Run pytest test suite:
   ```bash
   pytest tests/test_split_leakage.py
   ```
   - Passes with 0 failures.
   - Confirms case disjointness across all three splits.
   - Confirms every canonical class has at least 1 image and case in each split.
3. Check `ROADMAP.md`:
   - Ticket T01 and T02 are ready to be implemented and checked off in Section 9.
