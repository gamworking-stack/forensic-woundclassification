# Ticket T37: YOLO Wound Detector (Arm 4Y), Full Detection Evaluation, and the Vote with Detector Crops

- **Ticket ID**: `T37`
- **Phase**: Extra experiment on the T23 data (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T19](T19_auto_box_roi.md) (detector protocol, arms 4F and 4Z), [T23](T23_relabel_rerun.md) (data and runs), [T25](T25_patient_cases.md) (patients as cases), [T30](T30_more_contrast.md) (paired comparison), [T36](T36_fusion.md) (soft voting)
- **Scenarios**: none new. Classifiers are the T23 full-image (S01–S15) and per-crop (S25–S32) models, unchanged.
- **Status**: Running (decisions frozen 2026-11-15, before any YOLO run)

---

## 1. Goal

The researcher asked (2026-11-15):
- train a YOLO object detector for the wounds;
- report a full detection evaluation table: IoU, precision, recall, F1, AP50, AP75, mAP50–95, false positives per image, FROC, and sensitivity at a fixed false-positive rate;
- repeat the T36 vote between the full-image and crop models, with crops cut from the detector's boxes instead of the manual boxes.

T36 needed manual boxes. This ticket asks whether the vote still helps when a detector supplies the boxes, which is a fully automatic pipeline once the detector is trained.

## 2. Decisions (2026-11-15, before any YOLO run)

### 2.1 Detector: arm 4Y

| Item | Decision | Reason |
|---|---|---|
| Model | YOLO11 (ultralytics 8.4.178, `venv_exp4_yolo`) from COCO weights, sizes **s, m, l** = configurations `yolo11s`, `yolo11m`, `yolo11l` | One current single-stage detector family in three sizes. Weights are files in `/project/a260012_FORWCL/yolo_weights` (SHA-256 below) |
| Classes | One, `wound` (`single_cls`) | As every T19 arm; the classifiers name the wound type |
| Training data | Train split photos (728) and their unpadded manual boxes (`reference_boxes.csv`, 1,310 boxes); YOLO's own validation = the validation split (155 photos, 222 boxes). Photos EXIF-oriented and shrunk to a longest side of 1,600 px; test photos are never written to the YOLO dataset | The 4F data, unchanged |
| Recipe | `imgsz` 1280, at most 150 epochs, patience 30, batch 16, seed 42, deterministic, `optimizer=auto`, ultralytics' default augmentation (mosaic, HSV, scale, translate, left-right flip) plus up-down flip 0.5; `best.pt` = the epoch with the best validation fitness | Ultralytics defaults, plus vertical flips as in 4F (a wound photo has no "up") |
| Choice | `select.py --arm 4Y`: the size with the highest class-agnostic AP@0.5 on **validation**; ties go to the smaller model. Then the T19 F1 threshold and the **recall gate (≥ 0.50)** | Exactly as 4F |
| Detection | Photos shrunk as for training; confidence 0.001, YOLO NMS IoU 0.7, at most 300 boxes; then T19 step 1 (oversize rule, NMS 0.5, top 100), threshold, cap 53, fallback (top box, else whole photo) | T19 §2.3, so all arms share the same rules |
| Test | Detected **once**, with the chosen run, after `selection.json` exists (`test_allowed`) | T19 §2.6 |
| Settings | `YOLO_CONFIG_DIR=/project/a260012_FORWCL/yolo_config`: weights and runs on /project, **`sync` off** (no usage data sent to Ultralytics) | Nothing written into the repository; nothing leaves Raptor |

Weights (SHA-256): `yolo11s.pt` `85a76fe8…`, `yolo11m.pt` `d5ffc1a6…`, `yolo11l.pt` `9ebd0e09…`; `yolo26n.pt` `9b09cc8b…` is used only by ultralytics' AMP check.

### 2.2 Detection evaluation

`src/detection/detection_metrics.py`, on test, for 4Y and, for comparison, the T23 runs of 4F (Faster R-CNN, fine-tuned) and 4Z (Grounding DINO, zero-shot). The thresholds are each arm's frozen ones; nothing is re-chosen.

| Metric | Definition here |
|---|---|
| IoU | Mean IoU of matched pairs (IoU ≥ 0.5); and the mean over manual boxes of the best IoU any final box reaches (0 if none overlaps), which also counts misses |
| Precision, recall, F1 | Final boxes at the frozen threshold (cap and fallback applied), greedy one-to-one matching at IoU ≥ 0.5, pooled over boxes |
| AP50, AP75, mAP50–95 | COCO, 101-point, maxDets 100, on the step-1 candidates. One class, so mAP is that class's AP. AP by box size from image-share tertiles (T19) |
| False positives per image | (final boxes − matched) ÷ photos |
| FROC | Sensitivity against FP per image as the threshold sweeps the capped candidates (no fallback) |
| Sensitivity at fixed FP | The highest sensitivity reached at ≤ 1/8, 1/4, 1/2, **1**, 2, 4, 8 FP per image; the **FROC score** is their mean |
| Photo level | Photos with at least one / every wound found, photos with no box at the threshold |
| Intervals | 2,000 bootstrap resamples, seed 42, **patients** resampled (T25); overall and per wound type |

### 2.3 Vote with detector crops

| Item | Decision | Reason |
|---|---|---|
| Crops | 4Y final boxes on the 149 test photos, 8% padding (`create_auto_roi_dataset.py --splits test`). 4F crops are the existing T19 ones on the T23 data | Same cutting code as the manual crops |
| Classifier | The T23 per-crop models (S25–S32), unchanged, predict every auto crop (`predict_on_manifest.py --design crop`); `--check-manual` re-predicts their manual test crops first | No retraining: the question is the pipeline, not new models |
| Vote | T36 photo level, **equal weights** (w = 0.5): `0.5 × full image + 0.5 × mean of the photo's auto crops`. Also `crop_only` (mean of the auto crops alone), descriptive | T36 found that tuned weights add nothing. The crop level needs manual crops to score against, so it is not used |
| Comparisons | Paired patient bootstrap (T30 method), Holm over 8 families, per class: (a) auto vote − full image alone; (b) auto vote − manual-box vote (T36 equal); (c) auto crops alone − full image | (a) does the automatic pipeline help? (b) what do the detector's boxes cost? |
| Fallback photos | Kept: a photo with no box above the threshold uses its top box, and one with no box at all uses the whole photo as its "crop" | T19 rule; the test set stays 149 photos |

## 3. Files

| File | Change |
|---|---|
| `src/detection/yolo.py` | New: `prepare`, `train`, `detect` (arm 4Y) |
| `src/detection/detection_metrics.py` | New: full evaluation (`evaluate`) and arm comparison (`report`: table, per type, FROC and PR figure) |
| `src/detection/select.py` | Arm `4Y`, selected on validation |
| `src/data/create_auto_roi_dataset.py` | `--splits` (T37 cuts test only) |
| `src/evaluation/predict_on_manifest.py` | `--design crop` (per-crop models) |
| `src/evaluation/fusion.py` | `--splits`, `--rules` (adds `crop_only`), `--levels`; defaults unchanged (T36) |
| `tests/test_yolo_detection.py`, `tests/test_fusion.py` | YOLO labels, shrink and scale-back, test guard, 4Y selection, test-only crops; FROC and operating point by hand, seeded patient bootstrap, CLI and report; test-only photo-level vote |
| `scripts/slurm/t37.sbatch`, `scripts/slurm/t37_fusion.sbatch`, `scripts/hpc/submit_t37.sh` | Jobs, chained with dependencies |

## 4. Run log

- 2026-11-15 (laptop): `pytest tests/test_yolo_detection.py tests/test_fusion.py tests/test_auto_roi.py tests/test_detection_evaluation.py tests/test_detection.py tests/test_contrast_compare.py`: 56 passed (Python 3.11, CPU torch).

## 5. Results

Pending.
