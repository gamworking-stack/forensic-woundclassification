# Ticket T19: Auto-Box ROI Experiment (Experiment 4)

- **Ticket ID**: `T19`
- **Phase**: Extra experiment, outside the 16-scenario study (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T03](T03_cvat_roi_cropping.md) (crops); [T08](T08_evaluation_module.md) (metrics, case bootstrap); [T08b](T08b_lr_tuning.md) (tuning); [T12b](T12b_cross_validation.md) (folds); [T14](T14_gradcam.md) (Grad-CAM); [T15](T15_summary_and_full_vs_roi.md) (paired comparison); [T16](T16_final_verification.md) (verification); [T17](T17_box_roi.md), [T18](T18_crop_roi.md) (per-box and per-crop designs)
- **Scenarios**: S33–S104, reserved by detector arm. 4Z is S33–S56, 4P is S57–S80, 4F is S81–S104. Within an arm: union, then per-box, then per-crop, 8 families each in S01–S16 order.
- **Decision record**: the Wayfinder map `.scratch/auto-box-roi/map.md` and its tickets (one ticket per decision, with the reasoning)
- **Terms**: `GLOSSARY.md` (manual box, auto box, padding, wound detector, detector arm, cross-fitted auto box, ROI designs, wound types)
- **Status**: Planned. **Frozen 2026-10-16**, before any detector was run on our photos or any Experiment 4 result existed.

---

## 1. Goal

Experiments 1–3 crop the photo to **manual boxes** drawn in CVAT, and a deployed system would not have those (README limitation: "per-box crops still need manual boxes"). T19 replaces them with **auto boxes** from wound detectors and asks:

- **Primary**: can auto boxes replace manual boxes without a loss in classification? This is tested as **non-inferiority** of auto-box ROI against manual-box ROI, for the same trained models.
- **Secondary**:
  - Is auto-box ROI still better than the full image?
  - How do models retrained on auto-box crops compare?
  - How well do the detectors find the manually boxed wounds?

Three detector arms supply auto boxes:
- **4Z**: zero-shot, no training on our data.
- **4P**: a public wound model.
- **4F**: Faster R-CNN fine-tuned on the training split's manual boxes.

T19 changes nothing in the main, T17, T18 or CV runs; it has its own experiments roots. The test set is used once more, after every choice is fixed, and the thesis reports T19 as a post hoc analysis.

---

## 2. Decisions (2026-10-16, before any detector or classifier was run)

Decided with the researcher after the results of Experiments 1–3 were known. Each row links to the map ticket that holds the reasoning.

### 2.1 Data and reference boxes

| Item | Decision |
|---|---|
| Photos and split | The 1,100 photos and the frozen case-level split, unchanged: train 779 photos / 294 cases, validation 164 / 62, test 157 / 62. The 6 photos with no manual box stay excluded: there, "no box" means "not annotated", not "no wound". |
| Reference boxes | **Unpadded CVAT boxes** (1,667; train 1,177 / validation 277 / test 213), read from the 30 Sep 2026 CVAT exports. They are used for training 4F, for scoring every arm, and for the oversize rule. The manifests' `x1..y2` are **padded** crop coordinates and are not used as reference boxes. |
| Padding | 8% around every box, manual or auto, applied only when the crop is cut. |

### 2.2 Detector arms ([4Z setup](../.scratch/auto-box-roi/issues/04-zero-shot-setup.md), [4P choice](../.scratch/auto-box-roi/issues/05-public-model-choice.md), [4F recipe](../.scratch/auto-box-roi/issues/06-finetuned-detector-recipe.md))

| Arm | Candidates | Chosen by |
|---|---|---|
| **4Z** zero-shot | Grounding DINO (`IDEA-Research/grounding-dino-base`) and OWLv2 (`google/owlv2-large-patch14-ensemble`), safetensors only, loaded offline. Four prompts, frozen 2026-10-16: **P1** `wound .`; **P2** `injury on skin .`; **P3** the 7 class names; **P4** one phrase per wound type following the proposal's definitions (text in the 4Z ticket). Tiling (2 × 2 overlapping tiles plus the full image, merged by NMS) is pre-registered as an on/off option. 16 configurations. | Class-agnostic mAP@0.5 on **train + validation**, because 4Z trains nothing. Ties go to fewer prompt words, then no tiling. |
| **4P** public model | Hanterdsith YOLOv8 (`trapezius60/forensic_wound_detection`, revision `0c61ba14…`): `best.pt` (v8n), `best_yolov8m_roboflow_v7.pt`, `best_yolov8x_robov9.pt`, each on colour and on grayscale input. **WILLIE MINI** (`QianGroup/willie-weights`) only if it loads in one smoke test from the public code. Up to 7 configurations. All 8 Hanterdsith classes count as "wound". Segmentation-only foot-ulcer models are excluded. | Class-agnostic mAP@0.5 on **train + validation**. Ties go to the smaller model, then colour. |
| **4F** fine-tuned | `fasterrcnn_resnet50_fpn_v2` from COCO weights, 2 classes (background, wound), `min_size=1024`, `max_size=1365`. Horizontal and vertical flip, mild brightness/contrast jitter; no rotation, no random crop. SGD, momentum 0.9, weight decay 1e-4, batch 4, 500-iteration warmup, at most 60 epochs, patience 10, fp32, seed 42. Learning-rate grid {0.005, 0.01, 0.02}. | Checkpoint and learning rate by class-agnostic mAP@0.5 on **validation**. |

**4P checkpoint safety**: the Hanterdsith `.pt` files are full pickles. Each is scanned with `picklescan`, loaded once in a job whose only writable bind is a scratch folder, and re-saved as a plain `state_dict` or safetensors file. The SHA-256 of both the original and the re-saved file is recorded. Only the re-saved file is used afterwards.

**Zurich model** (Zimmermann et al. 2023): it was not requested before the freeze. If it arrives later, it is future work, not a T19 arm.

### 2.3 From detections to crops ([evaluation protocol](../.scratch/auto-box-roi/issues/07-detection-evaluation-protocol.md))

Every arm uses the same order:

1. **Raw boxes**: drop any box whose area share of the image exceeds that of the largest training-split manual box (unpadded); apply NMS at IoU 0.5; keep the top 100 by score. **mAP is computed here.**
2. **Operating threshold**: the score that maximises pooled box F1 at IoU ≥ 0.5 on the arm's selection data, counting the boxes with score ≥ threshold after the per-photo cap and **before any fallback**. The gate's recall (§2.4) is the recall at this threshold, computed the same way. That is validation for 4F, and train + validation for 4Z and 4P. Every distinct score is a candidate; ties go to the higher threshold.
3. **Final boxes**: apply the threshold, then the per-photo cap (the training-split maximum of manual boxes, 35). A photo with no box keeps its top-scoring box; with none at all, it falls back to the **full image**. Operating-point metrics are computed here.
4. **Crops**: union ROI (all final boxes merged into one crop) and per-box ROI (one crop per final box), each with 8% padding, exactly as T03/T17 cut them.

### 2.4 Gate ([retraining rules](../.scratch/auto-box-roi/issues/09-retraining-rules-and-budget.md))

An arm goes forward only if **recall at its operating threshold is ≥ 0.50** (pooled boxes, IoU ≥ 0.5) on its own selection data.

An arm that fails:
- makes no crops;
- is not retrained;
- **never touches test**;
- is reported as a negative finding, with its development detection metrics.

Thesis wording for a failed 4P: "No publicly available wound detector reached the pre-specified recall (≥ 0.50 at IoU ≥ 0.5) on the development photographs; arm 4P was therefore not carried forward."

### 2.5 4F boxes on its own training photos ([cross-fitting](../.scratch/auto-box-roi/issues/10-finetuned-train-split-boxes.md))

The main 4F detector was fitted to the training photos, so its boxes there would be too good. Training photos are therefore boxed by **5 cross-fitted 4F detectors**:

- **Groups**: the five groups of training cases from `data/splits/cv_folds.csv`, restricted to training cases. Fold sizes are 64 / 57 / 60 / 55 / 59 cases.
- **Training**: each detector learns from 4 groups and boxes only the fifth.
- **Recipe**: the main recipe, with the main detector's learning rate and best epoch count fixed.
- **Threshold**: the main 4F operating threshold.

Validation and test photos are boxed by the main 4F detector. Its validation boxes are mildly optimistic, because selection happened there; this is stated as a limitation.

4Z and 4P train nothing on our data, so one detector boxes every split.

### 2.6 Detection evaluation ([evaluation protocol](../.scratch/auto-box-roi/issues/07-detection-evaluation-protocol.md))

| Item | Decision |
|---|---|
| mAP | `pycocotools` COCO evaluation, one category, 101-point interpolation, maxDets 100. **mAP@0.5 is primary.** Also mAP@0.5:0.95 and AP for small / medium / large boxes. The size bins are **shares of the image area**, cut at the training-split tertiles of manual box share (`reference_limits.json`: 0.99% and 3.79%). COCO's pixel bins would put every manual box in "medium" or "large" (0 small, 44 medium, 1,623 large at source resolution). |
| Operating point | Greedy one-to-one matching by score at IoU ≥ 0.5. Reported: precision, recall, F1, mean IoU of matched pairs, and the share of photos with no auto box before the fallback. |
| Photo level (secondary) | Share of photos with at least one manual box found; share with every manual box found; mean per-photo recall. |
| By wound type | COCO mAP and recall on each type's photos. Precision for type X counts the auto boxes on photos of type X. |
| Intervals | Case-level bootstrap as in `src/evaluation/metrics.py`: 2,000 resamples, seed 42, 95% percentile intervals. Applied on test to mAP@0.5, recall, precision and the photo-level rates, overall and per type; small types are not merged. |
| When test is scored | **Once**, after every arm has fixed its model, prompt and threshold, in the same pass that cuts the test crops. |
| Selection reports | Every configuration of 4Z and 4P, and every 4F learning rate, on its selection data. For 4F, also the recall of the cross-fitted boxes on training photos next to the main detector's recall on validation. |

### 2.7 Classification analyses ([comparisons](../.scratch/auto-box-roi/issues/08-comparison-and-statistics.md))

| Item | Decision |
|---|---|
| Primary: existing models on auto-box crops | **union**: union-ROI models (even S02–S16) on auto union crops vs the same models on manual union crops. **per-box**: S17–S24 models on auto per-box crops vs manual per-box crops, at photo level as the mean of crop probabilities. Run for every arm that passes its gate, so at most 3 arms × 2 designs = 6 tests. |
| Primary test | **Non-inferiority, margin Δ = 0.03**, on the **pooled** (8-family mean) auto − manual test macro-F1. Δ is about half the CV ROI gain (+0.060), so the test asks whether auto boxes keep at least half of ROI's benefit. The result is **not inferior** if the lower 95% bound is above −0.03, **inferior** if the upper bound is below −0.03, and **inconclusive** otherwise. Holm over the passing arm × design tests, on one-sided p-values against −Δ. |
| Per family | The same difference for each family; Holm over 8 families within each arm × design (secondary). |
| Secondary, two-sided | (i) **Retrained** auto − manual, for the same designs. (ii) Auto ROI − full image (the full-image models, odd S01–S15). Holm over 8 families each; pooled effect reported. |
| Arms against each other | Descriptive only, no tests. |
| Statistics | Photo-level test macro-F1. Paired case bootstrap, 2,000 resamples, seed 42; pooled effect computed within each resample; two-sided p = min(1, 2 × min(P(d ≤ 0), P(d ≥ 0))). Accuracy, balanced accuracy and macro AUC are reported without tests. |
| Fallback photos | Kept in the main analysis, with the full image as the ROI model's input, so the test set stays 157 photos. Their count per arm is reported. A sensitivity analysis excluding them is descriptive only. |
| Per-crop design (Experiment 3 with auto boxes) | Results only, no comparison. Also reported: the share of auto crops that overlap no manual box. |

### 2.8 Retraining ([retraining rules](../.scratch/auto-box-roi/issues/09-retraining-rules-and-budget.md))

| Item | Decision |
|---|---|
| Scope | Every arm that passes its gate: union, per-box and per-crop, 8 families each, so 24 scenarios per arm and ≤ 72 in all. |
| Rules | Identical to S01–S32: lr grid {1e-5, 3e-5, 1e-4} for the six single networks; CNN-SVM 29-point grid; AlexNet ensemble seeds 43–45 at lr 3e-5; at most 200 epochs, patience 25, batch 32, seed 42. Selection unit: union and per-box at photo level, per-crop at crop level. |
| Crops | Auto-box crops in **every split**, built under the same threshold, cap and fallback rules. 4F training photos use cross-fitted boxes (§2.5). |
| Order | First the primary analysis (no training) for every passing arm. Then retraining in the order 4F → 4Z → 4P. |
| If GPU or time runs short | Cut per-crop retraining first, then the retraining of the passing arm with the lowest selection-data mAP@0.5. The primary analysis is never cut. Experiments 1–3 plus CV took about 24 GPU-hours, so this is not expected to be needed. |

### 2.9 Supplementary

| Item | Decision |
|---|---|
| Cross-validation | **None**: per-fold detectors would be needed, and the primary question is pre-specified on test. |
| Grad-CAM | The T14 method on the 14 T14 example photos, existing models, auto-box crops next to manual-box crops, for every passing arm and both primary designs (`gradcam.py`). Overlays stay on Raptor. |
| Verification | Required, T16-style (§3.8). |

### 2.10 Environment ([Raptor check](../.scratch/auto-box-roi/issues/03-raptor-environment.md))

| Item | Decision |
|---|---|
| Container | `/common/sif/pytorch-24.11-py3.sif` (torch 2.6.0a0 pre-release, torchvision 0.20.0a0, pycocotools, OpenCV 4.10, numpy 1.26.4). |
| Extra packages | Venvs on `/project`, created with `--system-site-packages`, never in `~/.local`. `venv_exp4` holds **transformers 4.52.4**; 4.57.x fails on this torch pre-release. `venv_exp4_yolo` holds **ultralytics 8.4.178**, using the container's numpy and OpenCV. |
| Certificates | Pass `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `PIP_CERT=/etc/ssl/certs/ca-certificates.crt` to the container. |
| Weights | `HF_HOME=/project/a260012_FORWCL/hf_cache`, runs with `HF_HUB_OFFLINE=1`. Grounding DINO snapshot `12bdfa31…`, OWLv2 snapshot `95e26936…`, Hanterdsith revision `0c61ba14…`. COCO Faster R-CNN weights go in `torch_cache`. |
| Verified | Grounding DINO and OWLv2 load offline on a compute node (CPU: 15 s and 51 s for one image). ultralytics imports. Its "torchvision 0.20 vs torch 2.6" warning is expected on this NVIDIA pairing and must be confirmed on the first YOLO run. |

---

## 3. Requirements

1. **Reference boxes** (`src/detection/boxes.py`):
   - read the unpadded CVAT boxes for the 1,100 photos and check them against the manifests (1,667 boxes, same photos and splits);
   - box helpers: IoU, NMS, the oversize rule, the per-photo cap, the fallback;
   - the largest training-split box share and box count, computed and written once.
2. **Detectors**:
   - `src/detection/zero_shot.py` (4Z), `src/detection/public.py` (4P), `src/detection/finetune.py` (4F training, including the cross-fitted runs).
   - Every arm writes one **detections file per configuration**, with one row per box: photo, score, x1..y2 in source pixels, and the source class (4P).
3. **Selection and gate** (`src/detection/select.py`): for each arm's selection data, it computes
   - mAP@0.5 per configuration and the chosen configuration;
   - the F1 threshold;
   - recall at that threshold, and the **gate verdict**.
   Output: `selection.json` per arm. Test photos must not be read here.
4. **Detection evaluation** (`src/detection/evaluate_detection.py`):
   - `pycocotools` mAP, the operating-point and photo-level metrics, the per-type breakdown, and the case bootstrap (§2.6);
   - run on test once per passing arm, from frozen `selection.json` files.
5. **Auto-box crops** (`src/data/create_auto_roi_dataset.py`):
   - final boxes become union and per-box crops, using the T03/T17 cropping code with 8% padding;
   - outputs: manifests `data/roi_auto/<arm>/{union,box}/roi_manifest.csv` (with a `fallback` column), plus `crop_summary.json`.
   - The same input paths (`roi`, `box`, `crop`) and the training code are reused, with new manifests and image roots.
6. **Primary analysis**:
   - `src/evaluation/predict_on_manifest.py`: load the existing union-ROI (even S02–S16) and per-box (S17–S24) checkpoints (CNN-SVM and the AlexNet ensemble included) and predict on auto crops.
   - `src/evaluation/auto_vs_manual.py`: the paired non-inferiority test with Holm across arms × designs, per-family results, the two-sided secondary comparisons, the fallback sensitivity analysis.
   - Outputs: `summary/` tables in the style of T15.
7. **Retraining**: `submit_all.sh` with new run roots `auto_<arm>`. `evaluate.py` names the scenarios S33–S104 according to §2.8.
8. **Verification** (`src/evaluation/verify_auto.py`):
   - reference boxes match the CVAT exports;
   - no test photo appears in any selection input;
   - each arm's selection and threshold are reproduced from its detections file;
   - every gate verdict is reproduced;
   - 4F training photos carry cross-fitted boxes only;
   - test sets are 157 photos per arm;
   - the 4P checkpoint hashes match;
   - every reported number is recomputed;
   - the retraining budget equals the main run's.
9. **Grad-CAM**: `gradcam.py` gains an auto-crop mode for the 14 T14 photos (§2.9).
10. **HPC**:
    - `scripts/slurm/detect.sbatch` and `scripts/slurm/train_detector.sbatch`;
    - `scripts/hpc/submit_auto.sh`, running in the order of §2.8;
    - all with the venv, certificate and offline settings of §2.10.
11. **Tests** (`tests/test_detection.py`, `tests/test_auto_roi.py`, CPU, synthetic, no downloads):
    - box helpers;
    - COCO mAP on hand-made cases;
    - the F1 threshold and its ties;
    - the gate;
    - the fallback;
    - the cross-fit split never boxes a photo with a detector that saw its case;
    - crops match T03 for identical boxes;
    - the non-inferiority verdicts (not inferior / inferior / inconclusive);
    - verification catches a test photo in selection data.

---

## 4. Files

| File | Change |
|---|---|
| `src/detection/__init__.py`, `boxes.py`, `zero_shot.py`, `public.py`, `finetune.py`, `select.py`, `evaluate_detection.py` | New |
| `src/data/create_auto_roi_dataset.py` | New |
| `src/evaluation/predict_on_manifest.py`, `auto_vs_manual.py`, `verify_auto.py` | New |
| `src/evaluation/evaluate.py`, `gradcam.py` | Scenarios S33–S104; auto-crop Grad-CAM mode |
| `scripts/slurm/detect.sbatch`, `train_detector.sbatch`, `scripts/hpc/submit_auto.sh` | New |
| `tests/test_detection.py`, `tests/test_auto_roi.py` | New |
| `requirements-exp4.txt` | `transformers==4.52.4`, `ultralytics==8.4.178`, `picklescan` (venvs only) |
| `runbook.md`, `ROADMAP.md`, `tickets/TICKET_MEMO.md`, `GLOSSARY.md` | Commands, scope, memo, terms |

---

## 5. Acceptance Check

1. `pytest tests/` passes, locally and in the Raptor container.
2. Every arm has a `selection.json` with its chosen configuration, threshold and gate verdict, written before any test detection exists. The file timestamps and the verification confirm the order.
3. For every passing arm:
   - detection metrics on test with intervals;
   - auto crops for all 157 test photos;
   - the primary non-inferiority result for union and per-box;
   - the secondary comparisons;
   - the retrained scenarios (S33–S104 for the passing arms) evaluated.
4. `verify_auto.py` reports 0 FAIL.
5. Results recorded in §6 of this ticket; `ROADMAP.md` §9 ticked.

---

## 6. Results

Not run yet.
