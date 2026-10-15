# Methodology: ROI Experiments for Forensic Wound Classification

This document records the methods of the three experiments in this project, in the order they were run.
- **Experiments 1 and 2** ask the same question: **does cropping to the wound region (ROI) improve deep-learning classification of forensic wound photographs over the full photograph?** They differ only in how the ROI crop is made.
- **Experiment 3** uses the per-box crops of Experiment 2 and reports how well each wound crop is classified on its own, with no comparison.

| | Experiment 1: union ROI | Experiment 2: per-box ROI | Experiment 3: per-crop ROI |
|---|---|---|---|
| ROI crop | All boxes of a photo merged into **one union crop** | **One crop per box**; a photo's prediction is the mean of its crops | The same crops as Experiment 2, **each crop its own sample** |
| Unit of evaluation | Photograph | Photograph | Crop |
| Scenarios | S01–S16 (8 families × full image / union ROI) | S17–S24, compared with the full-image scenarios of Experiment 1 | S25–S32, no comparison |
| Tickets | T01–T16 | T17 | T18 |
| Status | Main thesis study | Extra experiment, decided after Experiment 1's test results | Extra experiment, decided after Experiments 1 and 2's test results |

Details, commands and run logs are in the tickets (`tickets/`), the phase specs (`specs/`) and `runbook.md`. All numbers below come from the result files of the runs.

---

## 1. Shared foundation (all experiments)

### 1.1 Data

- **Source**: 1,100 forensic wound photographs from 418 cases (patients), university-affiliated hospital in Thailand, 2021–2026. All have at least one annotated wound box (§1.3).
- **Labels**: 7 classes, the dominant wound type from the case description, confirmed by expert review using DiMaio's definitions.

| Raw label | Class |
|---|---|
| `cut` | cut wound |
| `graze` | graze wound |
| `gsw` | gunshot wound (includes firearm grazes) |
| `contusion` | contusion |
| `laceration` | laceration |
| `scratch` | scratch |
| `penetrating wound (sharp&blunt)` | penetration wound |

- **Validation** (T01): required columns present, no missing values, every label maps to one of the 7 classes, no duplicate files. Result: 0 errors.

### 1.2 Case-level split (T02)

- Cases (`pseudonimized_id`) are split 70 / 15 / 15 into train, validation and test. All photographs of a case stay in one set, so no patient appears in two sets.
- Because a case can have more than one wound type, the assignment is chosen by a seeded random search (seed 42, 50,000 trials) that minimises class-proportion deviation and requires every class in every set.
- Result: 294 / 62 / 62 cases, 779 / 164 / 157 photographs. The split was frozen and every later step reads it unchanged.

### 1.3 Bounding-box annotation and matched scope (T03, T04)

- Wounds were annotated with bounding boxes in CVAT (7 YOLO 1.1 exports, one wound class per task).
- Both input paths use the same **1,100 annotated photographs** (418 cases; 779 / 164 / 157 photographs), so a full-image model and an ROI model differ only in the input.
- *Exclusion record*: 1,106 photographs from 421 cases were collected. 6 had no wound box and were excluded before any model was trained, which also removed 3 cases that had only that one photograph. The split was drawn on the 421 cases, seed 42, before the exclusion; the counts above are after it.
- The test set is **157 photographs from 62 cases** for every scenario in all experiments (213 crops from these photographs in Experiment 3).

### 1.4 Model families

| Family | Type |
|---|---|
| EfficientNet-B0 | CNN |
| ResNet152 | deep residual CNN |
| MobileNetV3-Small | lightweight CNN |
| GoogLeNet | Inception CNN |
| VGG16 | classic deep CNN |
| ViT-B/16 | vision transformer |
| CNN-SVM | frozen VGG16 + ResNet152 + EfficientNet-B0 features → SVM |
| AlexNet ensemble | 3 AlexNets, soft voting |

### 1.5 Training protocol (fine-tuned families, T05)

| Setting | Value |
|---|---|
| Weights | ImageNet-pretrained, all layers fine-tuned, new 7-class head |
| Preprocessing | EXIF orientation → RGB → pad to square (black) → resize to 384 × 384 (ViT-B/16: 224 × 224) → ImageNet normalisation |
| Augmentation (train only) | horizontal and vertical flips; rotation ±15°; random resized crop, scale 0.85–1.0; brightness/contrast jitter 0.2; Gaussian noise σ = 0.02 (p = 0.3); no hue/saturation change, because colour carries information for contusions |
| Loss | cross-entropy, label smoothing 0.05, class weights = inverse class frequency on the training set, normalised to mean 1 |
| Optimiser | AdamW, weight decay 1e-4, constant learning rate, batch 32 |
| Epochs | up to 200, early stopping with patience 25 on validation macro-F1 |
| Checkpoint | highest validation macro-F1 (ties: lower validation loss) |
| Precision, hardware | bf16 on NVIDIA H100 GPUs (MedCMU-HPC Raptor) |
| Seed | 42 (AlexNet ensemble members 43, 44, 45) |

**Learning-rate tuning** (T08b): each fine-tuned single family is trained with learning rates {1e-5, 3e-5, 1e-4}. The rate with the highest validation macro-F1 is kept, and only that run predicts the test set. The grid was not widened after results were seen.

**CNN-SVM** (T06):
- Frozen ImageNet features are taken from VGG16 (4,096), ResNet152 (2,048) and EfficientNet-B0 (1,280) and concatenated into 7,424 dimensions.
- The classifier is a standard scaler followed by an SVM with balanced class weights.
- The SVM setting is chosen on validation macro-F1 from 29 settings:
  - RBF kernel: C ∈ {0.1, 1, 10, 100, 1000} × gamma ∈ {scale, 1e-6, 1e-5, 1e-4, 1e-3};
  - linear kernel: C ∈ {0.001, 0.01, 0.1, 1}.
- Probabilities come from Platt scaling. For the test set, the chosen SVM is refit on train plus validation.

**AlexNet ensemble** (T07): three AlexNets are trained with seeds 43, 44 and 45 at learning rate 3e-5. Each member keeps its own best checkpoint. The ensemble probability is the mean of the three members' probabilities, so no combination weight is tuned.

### 1.6 Evaluation (T08, T15)

- **Primary metric**: test macro-F1, so every wound type counts equally.
- **Secondary metrics**: accuracy, balanced accuracy, macro precision and recall, one-vs-rest macro AUC, per-class precision, recall, F1 and AUC, and confusion matrices.
- **Confidence intervals**: case-level bootstrap, resampling cases with replacement and keeping all photographs of a case together. 2,000 resamples, seed 42, 95% percentile interval.
- **Paired comparison** (ROI − full image):
  - Each resample scores both inputs and all 8 families on the same cases, so each difference is paired by case.
  - The two-sided bootstrap p-value is min(1, 2 × min(P(d ≤ 0), P(d ≥ 0))).
  - P-values are Holm-adjusted over the 8 families, separately for each metric.
  - The pooled effect is the mean of the 8 family differences, computed within each resample.
- **The test set is used once per scenario**, after all choices were made on validation.

### 1.7 Supplementary cross-validation (T12b method)

- **Folds**: the 356 development cases (train plus validation, 943 photographs) are split into 5 case-grouped folds (seed 42, 50,000 trials, every class in every fold). The folds are frozen in `data/splits/cv_folds.csv`.
- **Training per fold**: each scenario trains on 4 folds and predicts the held-out fold. It reuses its main-run settings without re-tuning: the chosen learning rate, the chosen SVM, and the ensemble seeds. Training runs for a fixed number of epochs equal to the main run's best epoch. The test set is never loaded.
- **Outputs**:
  - mean ± SD of macro-F1 over the 5 folds;
  - out-of-fold predictions for all 356 cases, compared with the same paired bootstrap as §1.6.
- **Caveat**: the settings were chosen on the main validation split, which lies inside these folds. Absolute CV scores are therefore slightly optimistic. ROI − full differences are less affected, because both inputs share this.

### 1.8 Grad-CAM (T14 method)

- **Images**: one fixed set of 14 test photographs (2 per class, 14 different cases, seed 42), drawn before any map was computed and used for every family and input.
- **Map**: Grad-CAM for the predicted class, in fp32. Target layers:
  - EfficientNet-B0 and MobileNetV3-Small: `features[-1]`
  - ResNet152: `layer4[-1]`
  - GoogLeNet: `inception5b`
  - VGG16: `features[29]`
  - AlexNet (one member): `features[11]`
  - ViT-B/16: the last encoder block, tokens reshaped to 14 × 14
- **CNN-SVM** is excluded, because it has no gradient path to a single image map.
- **Patient images**: the overlays stay on the HPC.

### 1.9 Verification (T16 method)

An automated script re-checks the run from its files:
- every expected file exists;
- the test set equals the frozen split;
- each selection is reproduced from the validation scores;
- the training budget is equal across scenarios;
- every reported number is recomputed;
- the Grad-CAM predictions match the prediction files;
- the cross-validation covers every development case once.

---

## 2. Experiment 1: full image vs union ROI (S01–S16)

### 2.1 ROI crops

- When a photograph has several boxes, the boxes are merged into one **union box**: the smallest box containing all of them.
- The box is padded by 8% of its width and height on each side and clamped to the image. Crops smaller than 8 px are rejected, and crops are saved as JPEG at quality 95.
- Result: **1,100 union crops**, one per photograph (779 / 164 / 157). Each crop inherits its photograph's case and split.

### 2.2 Scenarios

| Family | Full image | Union ROI |
|---|---|---|
| EfficientNet-B0 | S01 | S02 |
| ResNet152 | S03 | S04 |
| MobileNetV3-Small | S05 | S06 |
| GoogLeNet | S07 | S08 |
| VGG16 | S09 | S10 |
| ViT-B/16 | S11 | S12 |
| CNN-SVM | S13 | S14 |
| AlexNet ensemble | S15 | S16 |

Each pair uses the same 1,100 photographs, split, seed, budget and tuning rules. The only difference within a family is the input.

### 2.3 Order of decisions

1. The training protocol and learning-rate grid were fixed before training (T05, T08b).
2. The plan for the paired full-vs-ROI analysis was fixed before any paired result was computed (T15 §2). The unpaired test metrics had already been seen at that point.
3. The cross-validation (§1.7) was decided **after** the test results, so it is reported as post hoc and supplementary (T12b §2).

### 2.4 Results summary

| | ROI − full macro-F1 (pooled, 8 families) | Families significant after Holm |
|---|---|---|
| Test (62 cases) | +0.036 [−0.031, +0.091], p = 0.288 | 0 of 8 |
| Cross-validation (356 cases) | +0.061 [+0.036, +0.085], p < 0.001 | 3 of 8 (CNN-SVM, MobileNetV3-Small, ViT-B/16) |

- Best test model: ResNet152 on the full image, macro-F1 0.783.
- Verification: 313 checks passed, 0 failed.

---

## 3. Experiment 2: full image vs per-box ROI (S17–S24)

### 3.1 Why and when

- Experiment 1 merged every box of a photograph into one crop, so a crop of a multi-wound photograph contains several wounds and the skin between them.
- Experiment 2 crops each annotated wound on its own.
- It was decided **after** Experiment 1's test results. Its design decisions (§3.2–3.5) were fixed before any per-box model was trained (T17 §2).

### 3.2 Per-box crops

- `create_roi_dataset.py --crop-mode individual`: each box is cropped separately with the same 8% padding, 8 px minimum size and JPEG settings as Experiment 1. No union is formed.
- Result: **1,667 crops** from the same 1,100 photographs and 418 cases (train 1,177 / validation 277 / test 213).
  - Photographs have up to 35 boxes.
  - 30 of the 157 test photographs have more than one box. For the other 127, the per-box crop equals the union crop.
- **Label**: a crop takes its photograph's label. Each CVAT task holds one wound class, so every box of a photograph was drawn under that class.
- **Split**: a crop inherits its photograph's case and split, and the leakage check runs on cases.

### 3.3 Training changes for per-box crops

Everything in §1.5 applies, with these changes:

| Item | Per-box rule | Why |
|---|---|---|
| Loss weight per crop | 1 / (training crops of its photograph) | Each photograph counts once, so photographs with many boxes do not dominate |
| Class weights | Computed from the weighted counts; this equals photographs per class, the same as Experiment 1 | Same class balance as the union path |
| CNN-SVM | Scaler and SVM fitted with the same per-crop sample weights; balanced class weights on weighted counts | Same weighting as the fine-tuned families |
| AlexNet ensemble | Soft vote per crop, then the photograph mean (§3.4) | Same combination rule |
| Model selection | Early stopping and the learning-rate choice use **photograph-level** validation macro-F1 (ties: lower crop-level validation loss) | Selection uses the primary unit |

The learning-rate grid, SVM grid, ensemble seeds, epochs and patience are the same as in Experiment 1.

### 3.4 Photograph-level prediction

- Every crop of a photograph is scored.
- The photograph's class probabilities are the **mean of its crops' probabilities**, and the predicted class is their argmax.
- This gives one prediction per photograph, so per-box models are scored on exactly the same 157 test photographs as the full-image models.
- Crop-level test metrics (213 crops) are reported descriptively only.

### 3.5 Scenarios and comparisons

| Family | Full image (from Exp. 1) | Per-box ROI |
|---|---|---|
| EfficientNet-B0 | S01 | S17 |
| ResNet152 | S03 | S18 |
| MobileNetV3-Small | S05 | S19 |
| GoogLeNet | S07 | S20 |
| VGG16 | S09 | S21 |
| ViT-B/16 | S11 | S22 |
| CNN-SVM | S13 | S23 |
| AlexNet ensemble | S15 | S24 |

- **Comparisons**: per-box − full image, and per-box − union ROI.
  - The T17 plan names per-box − union as the primary comparison. The thesis and AFSN presentations report per-box − full image, which answers the main research question.
  - Both use the paired case-level bootstrap with Holm correction from §1.6, on photograph-level test macro-F1.
- **Secondary metrics**: balanced accuracy and macro AUC.

### 3.6 Supplementary analyses (decided after the per-box test results; post hoc)

- **Cross-validation**: the §1.7 method with the frozen folds and each S17–S24 scenario's chosen settings as fixed epochs. Held-out crops are predicted and then averaged per photograph. The out-of-fold photographs (356 cases, 943 photographs) are compared with Experiment 1's out-of-fold predictions for the full image and the union ROI.
- **Grad-CAM**: the §1.8 method on every crop of the same 14 example photographs, with the checkpoint behind each S17–S24 test file. There is no "map inside the box" score, because a per-box crop *is* the box.
- **Verification**: the §1.9 checks for S17–S24 (`src/evaluation/verify_box.py`), with selection reproduced at photograph level.

### 3.7 Results summary

**Test set, photograph level (157 photographs, 62 cases)**

| Comparison | Pooled macro-F1 difference [95% CI] | Families significant after Holm |
|---|---|---|
| Per-box − full image | +0.073 [+0.009, +0.129], p = 0.028 | 0 of 8 (7 of 8 higher) |
| Per-box − union ROI | +0.037 [+0.006, +0.071], p = 0.027 | 1 of 8 (EfficientNet-B0) |

- Best test model across Experiments 1 and 2: EfficientNet-B0 on per-box ROI, macro-F1 0.811 [0.703, 0.878], accuracy 87.9%, macro AUC 0.969.

**Cross-validation, out of fold (356 cases, 943 photographs)**

| Comparison | Pooled macro-F1 difference [95% CI] | Families significant after Holm |
|---|---|---|
| Per-box − full image | +0.060 [+0.037, +0.083], p < 0.001 | 4 of 8 (CNN-SVM +0.228, MobileNetV3-Small +0.104, AlexNet ensemble +0.077, EfficientNet-B0 +0.061) |
| Per-box − union ROI | −0.001 [−0.018, +0.017], p = 0.903 | 0 of 8 |

**Grad-CAM**: the recomputed predictions matched the prediction files for 109 of 112 crops (96 of 98 photographs). The 3 differences are borderline MobileNetV3-Small crops, explained by fp32 here vs the bf16 used in training.

**Verification**: 203 checks passed, 0 failed, including the cross-validation, with 2 notes: all 21 training jobs ran from one code version, and the Grad-CAM matches above.

### 3.8 How to read Experiment 2

- Both experiments find that ROI cropping beats the full image on average:
  - Experiment 1 cross-validation: +0.061;
  - Experiment 2 test set: +0.073;
  - Experiment 2 cross-validation: +0.060.
- Per-box crops beat the union crop on the test set (+0.037), but **not in cross-validation (−0.001)**. The test-set advantage of per-box over union is therefore not confirmed; it may be chance in 62 test cases. Treat the two crop designs as performing about the same.
- The per-box design was chosen after the union design had been tested on the same test set, so the test-set results of Experiment 2 are not fully independent of that choice. The cross-validation, which never touches the test set, is the stronger evidence.
- In the test set, only 30 of the 157 photographs have more than one box. Any per-box effect can come from scoring multi-box photographs as the mean of their wounds, or from training on individual wound crops (1,177 instead of 779). This design cannot separate the two.

---

## 4. Experiment 3: per-crop ROI (S25–S32)

### 4.1 Why and when

- Experiment 2 scored each photograph as the mean of its crops. Experiment 3 asks how well a model classifies **a single wound crop**, with every crop a sample in training, model selection and evaluation.
- It was decided after the test results of Experiments 1 and 2 had been seen. Its design (§4.2–4.4) was fixed with the researcher before any per-crop model was trained (T18 §2).

### 4.2 Data

- The same 1,667 per-box crops as Experiment 2 (§3.2): train 1,177 / validation 277 / test 213, from 1,100 photographs and 418 cases.
- Each crop takes its photograph's label and split.

### 4.3 Training and selection

Everything in §1.5 applies, with these differences from Experiment 2:

| Item | Per-crop rule |
|---|---|
| Loss weight | None: every crop counts once |
| Class weights | Inverse class frequency of the **training crops**, normalised to mean 1 |
| CNN-SVM | Unweighted scaler and SVM, balanced class weights on crop counts |
| Model selection | Early stopping and learning rate on **crop-level** validation macro-F1 (277 crops); ties go to the lower validation loss |

The learning-rate grid, SVM grid, ensemble seeds, epochs and patience are the same as in Experiments 1 and 2.

### 4.4 Evaluation

- **Primary**: test macro-F1 per crop (213 crops, 62 cases), with a 95% case-level bootstrap interval. 2,000 resamples of cases keep all crops of a case together; seed 42.
- **Secondary**: accuracy, balanced accuracy, macro precision and recall, macro AUC, per-class precision, recall, F1 and AUC, and confusion matrices.
- **No comparison**: there is no photograph level and no paired test against the full image or the other ROI designs.
- **Supplementary**: verification only (`verify_crop.py`). There is no cross-validation and no new Grad-CAM.
- **Weighting caveat**: a photograph with several wounds counts once per crop. 30 of the 157 test photographs give more than one crop; for example, gunshot wound has 73 test crops from 17 cases.

### 4.5 Results summary

| Scenario | Family | Accuracy | Balanced acc. | Macro-F1 [95% CI] | Macro AUC |
|---|---|---|---|---|---|
| S25 | EfficientNet-B0 | 0.873 | 0.841 | **0.797** [0.707, 0.854] | 0.962 |
| S26 | ResNet152 | 0.864 | 0.804 | 0.767 [0.677, 0.839] | 0.952 |
| S27 | MobileNetV3-Small | 0.826 | 0.739 | 0.730 [0.608, 0.786] | 0.965 |
| S28 | GoogLeNet | 0.845 | 0.769 | 0.719 [0.638, 0.778] | 0.947 |
| S29 | VGG16 | 0.859 | 0.745 | 0.727 [0.626, 0.799] | 0.963 |
| S30 | ViT-B/16 | 0.845 | 0.721 | 0.715 [0.612, 0.782] | 0.954 |
| S31 | CNN-SVM | 0.812 | 0.734 | 0.694 [0.587, 0.787] | 0.972 |
| S32 | AlexNet ensemble | 0.840 | 0.776 | 0.737 [0.635, 0.809] | 0.947 |

- **Best model**: EfficientNet-B0, macro-F1 0.797, followed by ResNet152 at 0.767. The other six models are 0.69–0.74, with overlapping intervals.
- **Easiest classes**: gunshot wound, graze wound and contusion crops (mean F1 over the 8 families 0.93, 0.93 and 0.92).
- **Hardest classes**: laceration 0.74, scratch 0.60, cut wound 0.55 and penetration wound 0.49. Scratch and penetration have only 6 and 7 test crops, so one crop moves their F1 by about 0.1–0.2.
- **Verification**: 157 checks passed, 0 failed, with 1 note: all 21 trained networks ran from one code version.

---

## 5. Limitations common to all experiments

- **Data source**: single hospital, internal validation only, no external test set.
- **Test set size**: 62 test cases, with only 5 scratch and 7 penetration test photographs. Per-family intervals are about ±0.12–0.13 wide.
- **Same test set**: the test set has been used for all three ROI analyses.
- **Variation not captured**: one training seed per setting; the bootstrap resamples cases, not training runs.
- **Tuning grids**: several chosen learning rates and the CNN-SVM settings lie at the edge of their grids. The grids were not widened after results.
- **Manual boxes**: ROI crops depend on hand-drawn boxes, so a deployed system would need an automatic wound detector.
- **No subgroup analysis**: skin tone and autopsy status are not in the metadata.

---

## 6. Where things are

| Item | Location |
|---|---|
| Frozen split, CV folds | `data/splits/` |
| Union crop manifest | `data/roi/roi_manifest.csv` |
| Per-box crop manifest | `data/roi_box/roi_manifest.csv` |
| Training code | `src/training/` (`photo.py` holds the per-box weighting and photograph mean) |
| Evaluation code | `src/evaluation/` (`summarize.py`, `box_summary.py`, `cv_summary.py`, `cv_box_summary.py`, `crop_summary.py`, `gradcam.py`, `verify.py`, `verify_box.py`, `verify_crop.py`) |
| HPC results | `/project/a260012_FORWCL/experiments/{main,cv,roi_box,cv_box,roi_crop}` on Raptor |
| Aggregate results (laptop) | `../results_figures/{main,roi_box,cv_box,roi_crop}` (not in git) |
| Decision logs | `tickets/T01`–`T16` (Experiment 1), `tickets/T17_box_roi.md` (Experiment 2), `tickets/T18_crop_roi.md` (Experiment 3) |
