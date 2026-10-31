# Ticket T30: "More Contrast" Experiment (CLAHE-Enhanced Images, Retrained)

- **Ticket ID**: `T30`
- **Phase**: Extra experiment on the T23 dataset (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T23](T23_relabel_rerun.md) (data, split, runs), [T25](T25_patient_cases.md) (patient intervals), [T28](T28_precision_recall.md), [T29](T29_confidence_threshold.md)
- **Scenarios**: S01–S15 (full image) and S25–S32 (per-crop) retrained and tested on contrast-enhanced images; the T23 runs are not touched
- **Status**: Planned and submitted 2026-10-31

---

## 1. Goal

The researcher asked (2026-10-31) to add more contrast to every image and rerun the experiment, named "more contrast experiment". Decisions (researcher, 2026-10-31): retrain (not test-only), CLAHE, full image and per-crop, reported with the same metrics and the 0.60 confidence threshold (T29).

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Enhancement | CLAHE on CIELAB L* only (OpenCV 4.10, clip limit 2.0, 8 × 8 tiles); a* and b* unchanged | Researcher's choice; local contrast without changing wound colour |
| Where | Once per photograph, after the EXIF orientation, before anything else; saved as JPEG (quality 95, no chroma subsampling, no EXIF) at full size in `$DERIVED_DIR/raw_clahe` (`src/data/create_contrast_dataset.py`) | Every reader sees the same upright image at the same size |
| Crops | Cut from the enhanced photographs by the T03 crop job with the T23 per-box boxes and padding, checked against the committed T23 `roi_box/roi_manifest.csv` | Same crops, only the pixels differ |
| Data, split, boxes | T23 (`WOUND_DATA_DIR=data/relabel_2026_10_23`), unchanged | Only contrast differs from T23 |
| Training | Identical to T23: same families, learning-rate grid and selection on validation, seeds, augmentation (which still includes ±0.2 contrast jitter), epochs | One change at a time |
| Runs | `experiments_relabel_2026_10_23_more_contrast/{main,roi_crop}`, crops in `derived_data/relabel_2026_10_23_more_contrast/`; `scripts/hpc/contrast_env.sh` sets every folder | Own folders, as T23 |
| Code change | `train_single.sbatch` reads `RAW_IMAGE_ROOT` (default: the raw photographs) | Full-image runs need another photo folder |
| Reporting | Table 2 metrics with patient-level intervals (T25, T28), the 0.60 threshold (T29), and CLAHE minus T23 per scenario on the same test images | Same metrics as before, plus the comparison asked for |

## 3. Files

| File | Change |
|---|---|
| `src/data/create_contrast_dataset.py` | New: CLAHE copy of an image tree, `contrast_manifest.csv`, `contrast_report.json` |
| `tests/test_contrast_dataset.py` | Contrast up and colour kept; tree mirrored upright without EXIF and found by `find_source_image`; output inside the source refused |
| `scripts/slurm/contrast_images.sbatch` | New CPU job |
| `scripts/hpc/contrast_env.sh` | New |
| `scripts/slurm/train_single.sbatch` | `RAW_IMAGE_ROOT` |
| `runbook.md` (T23 step 13), `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Commands, scope, memo |

## 4. Run log

- 2026-10-31: `pytest tests/test_contrast_dataset.py` 3 passed (laptop, Python 3.11, OpenCV 4.10).
