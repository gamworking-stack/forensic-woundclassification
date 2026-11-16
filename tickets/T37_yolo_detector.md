# Ticket T37: YOLO Wound Detector (Arm 4Y), Full Detection Evaluation, and the Vote with Detector Crops

- **Ticket ID**: `T37`
- **Phase**: Extra experiment on the T23 data (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T19](T19_auto_box_roi.md) (detector protocol, arms 4F and 4Z), [T23](T23_relabel_rerun.md) (data and runs), [T25](T25_patient_cases.md) (patients as cases), [T30](T30_more_contrast.md) (paired comparison), [T36](T36_fusion.md) (soft voting)
- **Scenarios**: none new. Classifiers are the T23 full-image (S01–S15) and per-crop (S25–S32) models, unchanged.
- **Status**: Done (decisions frozen 2026-11-15 before any YOLO run; results 2026-11-16, §5)

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

- 2026-11-15 (Raptor, commit `c74a4e6`): `submit_t37.sh`, jobs 247206–247219. YOLO dataset 728 train photos / 1,310 boxes, 155 validation / 222. Training: yolo11s 13 min, m 26 min, l 33 min (H100). Selection, test detection, metrics (4Y, 4F, 4Z), report and 4Y crops completed.
- Job 247217 (4Y crop predictions) failed: `load_manifest` requires every class in every split, and the 4Y manifest holds test only. Fixed in `ce0be57` (`read_auto_manifest`: a test-only manifest is checked for all 7 classes among its test rows); resubmitted as 247538 (predictions) and 247539 (vote). `--check-manual`: all 8 per-crop models reproduced their reported manual test predictions (same class 1.0 on 276 crops; max probability difference < 1e-7).
- `3e55743`: precision-recall legend moved clear of the curves; report regenerated.
- Fetched (summaries and figures only, no per-photo rows) to `../results_figures/YOLO detection/`: `detection_t37/` (metrics.json per arm, report table, per type, `froc_pr.png`), `auto_4Y/` (selection, crop summary, YOLO11m run config and epoch log), `compare/` (6 comparisons).

## 5. Results (test: 149 photographs, 276 manual boxes, 55 patients; 95% patient-bootstrap CI)

### 5.1 Selection (validation)

AP@0.5 yolo11s 0.763, **yolo11m 0.783**, yolo11l 0.752: YOLO11m chosen. Threshold 0.367; recall at the threshold passed the gate (≥ 0.50). Test crops: 255 from 149 photos; 21 photos (14%) fell back to their top box, none to the whole photo.

### 5.2 Detection (`detection_metrics_table.csv`)

| Metric | YOLO11m (4Y) | Faster R-CNN (4F) | Grounding DINO (4Z) |
|---|---|---|---|
| Threshold (chosen on validation / train+val) | 0.367 | 0.930 | 0.286 |
| IoU of matched boxes | 0.807 [0.789, 0.824] | 0.809 [0.795, 0.823] | 0.780 [0.766, 0.795] |
| Best IoU per manual box (misses = 0) | 0.569 [0.484, 0.647] | 0.559 [0.481, 0.635] | 0.438 |
| Precision | 0.725 [0.659, 0.793] | 0.763 [0.703, 0.815] | 0.671 [0.593, 0.735] |
| Recall (sensitivity) | 0.670 [0.571, 0.765] | 0.652 [0.558, 0.750] | 0.547 [0.437, 0.656] |
| F1 | 0.697 [0.616, 0.771] | 0.703 [0.629, 0.772] | 0.603 [0.506, 0.684] |
| AP50 | 0.720 [0.617, 0.808] | 0.719 [0.616, 0.807] | 0.600 [0.491, 0.703] |
| AP75 | 0.461 [0.364, 0.552] | 0.446 [0.355, 0.532] | 0.288 [0.216, 0.372] |
| mAP50–95 | 0.421 [0.347, 0.491] | 0.425 [0.355, 0.489] | 0.307 [0.251, 0.364] |
| False positives per image | 0.470 [0.336, 0.619] | 0.376 [0.282, 0.486] | 0.497 [0.394, 0.606] |
| Sensitivity at 0.5 / **1** / 2 FP per image | 0.699 / **0.761** [0.657, 0.855] / 0.812 | 0.659 / **0.725** [0.625, 0.818] / 0.797 | 0.540 / **0.605** [0.487, 0.718] / 0.667 |
| FROC score (mean, 1/8–8 FP per image) | 0.717 [0.614, 0.805] | 0.717 [0.612, 0.807] | 0.600 |
| Photos with ≥ 1 / every wound found | 0.832 / 0.631 | 0.852 / 0.624 | 0.745 / 0.517 |

- **YOLO11 and Faster R-CNN detect equally well**: AP50 0.72 vs 0.72, FROC score 0.72 vs 0.72, all intervals overlapping. YOLO finds slightly more wounds at its threshold (recall 0.67 vs 0.65) with more false alarms (0.47 vs 0.38 per photo); at 1 FP per photo it finds 76% vs 73%. Faster R-CNN is better at very low false-alarm rates (≤ 1/4 FP per photo). Zero-shot Grounding DINO is clearly worse (AP50 0.60).
- **By wound type (YOLO AP50)**: gunshot 0.94, penetration 0.83, graze 0.72, laceration 0.72, cut 0.68, contusion 0.53, scratch 0.27 (12 photos, 6 patients). Contusions and scratches are hard for every detector.
- These T23 numbers are lower than T19's original-label 4F (AP50 0.868): the 23 Oct annotations split many wounds into more, smaller boxes (1,972 vs 1,677).

### 5.3 Vote with detector crops (photo level, equal weights; mean of 8 families)

| Comparison | Accuracy | Macro-F1 |
|---|---|---|
| **YOLO vote − full image** | 0.791 → 0.846, **+0.055 [+0.031, +0.081]** | 0.683 → 0.738, **+0.055 [+0.018, +0.098]** |
| Faster R-CNN vote − full image | 0.791 → 0.845, **+0.054 [+0.024, +0.084]** | 0.683 → 0.744, **+0.062 [+0.014, +0.094]** |
| YOLO vote − manual-box vote (T36) | 0.862 → 0.846, −0.016 [−0.048, +0.014] | 0.777 → 0.738, −0.039 [−0.082, +0.016] |
| Faster R-CNN vote − manual-box vote | 0.862 → 0.845, −0.018 [−0.034, −0.003] | 0.777 → 0.744, −0.033 [−0.072, −0.003] |
| YOLO crops alone − full image | 0.791 → 0.799, +0.008 [−0.035, +0.050] | +0.007 [−0.077, +0.084] |
| Faster R-CNN crops alone − full image | 0.791 → 0.823, +0.032 [−0.017, +0.078] | +0.041 [−0.060, +0.122] |

YOLO vote accuracy per family (full image → vote; * Holm p < 0.05): EfficientNet-B0 0.866 → 0.879; ResNet152 0.893 → 0.879; MobileNetV3-Small 0.745 → 0.812; GoogLeNet 0.826 → 0.872; VGG16 0.819 → 0.879; ViT-B/16 0.792 → 0.859*; CNN-SVM 0.651 → 0.772*; AlexNet ensemble 0.738 → 0.819.

- **The vote still helps without manual boxes.** With YOLO's boxes it keeps about three-quarters of the T36 gain (+0.055 of +0.071 accuracy); the remaining loss against manual boxes (−0.016) is not significant for YOLO, and small but significant for Faster R-CNN (−0.018).
- **The vote matters more than the detector**: auto crops alone are no better than the full image (+0.008 YOLO, +0.032 Faster R-CNN, both n.s.), but averaging them with the full image gives +0.055. The two inputs make different mistakes.
- **Per class** (YOLO vote − full image, pooled recall): graze +0.088 [+0.049, +0.131], laceration +0.094 [+0.048, +0.154] (both Holm-significant), gunshot +0.055 [+0.008, +0.114].
- As in T36, the vote lifts the weaker models (CNN-SVM +0.121, AlexNet ensemble +0.081, ViT +0.067); the best model (ResNet152, 0.893) does not improve (0.879).
- **Limits**: the test set is used once more; one detector seed; the YOLO threshold and size were chosen on validation, which also chose the classifiers' settings; the detector still needs manual boxes on the training photos.
