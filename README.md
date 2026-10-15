# Forensic Wound Classification

Deep-learning classification of **7 forensic wound types** from photographs, and a test of whether **cropping to the wound region (ROI)** improves it over the full photograph.

Master's thesis project, Preenapun Saokhieo, Chiang Mai University. Data: 1,100 forensic wound photographs from 418 cases at a university-affiliated hospital in Thailand, 2021–2026.

**Classes**: cut wound, graze wound, gunshot wound, contusion, laceration, scratch, penetration wound.

---

## Research question and design

> Does wound localization, in the form of bounding-box ROI crops, improve classification over full images?

- **8 model families**: EfficientNet-B0, ResNet152, MobileNetV3-Small, GoogLeNet, VGG16, ViT-B/16, CNN-SVM (frozen features → SVM) and an AlexNet soft-voting ensemble.
- **Fair comparison within each family**: every family is trained on both the full image and the ROI crop, with the same images, split, seed, training budget and tuning rules. The only difference is the input.
- **No patient leakage**: there is one frozen case-level split (70/15/15; 294 / 62 / 62 cases), so no patient is in two sets.
- **Single test use**: each scenario predicts the test set once, after selection on validation. The test set is 157 photographs from 62 cases for every scenario.
- **Statistics**:
  - Primary metric: test macro-F1.
  - Intervals: case-level bootstrap (2,000 resamples).
  - Comparisons: paired ROI − full image differences with Holm correction over the 8 families.
- **Supplementary checks**: 5-fold case-grouped cross-validation on the 356 development cases, decided post hoc; Grad-CAM; automated verification of every reported number.

## Three experiments

| | Experiment 1: union ROI | Experiment 2: per-box ROI | Experiment 3: per-crop ROI |
|---|---|---|---|
| ROI crop | All boxes of a photo merged into one crop (8% padding) | One crop per box (8% padding); photo prediction = mean of its crops | The same crops, each crop its own sample; scored per crop |
| Scenarios | S01–S16 (full + union ROI) | S17–S24 (per-box ROI), compared with the full-image scenarios of Experiment 1 | S25–S32, results only (no comparison) |
| Crops | 1,100 | 1,667 from the same 1,100 photos | the same 1,667 (213 test crops) |
| Plan and log | `tickets/T01`–`T16` | `tickets/T17_box_roi.md` | `tickets/T18_crop_roi.md` |

Full methodology: [docs/methodology_roi_experiments.md](docs/methodology_roi_experiments.md).

## Main results

| Comparison (macro-F1, pooled over 8 families) | Test set (62 cases) | Cross-validation (356 cases) |
|---|---|---|
| Union ROI − full image | +0.036 [−0.031, +0.091] | +0.061 [+0.036, +0.085] |
| Per-box ROI − full image | +0.073 [+0.009, +0.129] | +0.060 [+0.037, +0.083] |
| Per-box ROI − union ROI | +0.037 [+0.006, +0.071] | −0.001 [−0.018, +0.017] |

- **ROI improves on the full image on average.** Both crop designs show a gain in cross-validation, and the per-box design also shows one on the test set. The largest gains are for CNN-SVM and MobileNetV3-Small.
- **The two crop designs perform about the same.** Per-box beat the union crop on the test set, but not in cross-validation.
- **Best test model**: EfficientNet-B0 on per-box ROI, with macro-F1 0.811, accuracy 87.9% and macro AUC 0.969. The best full-image model is ResNet152, with macro-F1 0.783.
- **Per-crop classification (Experiment 3)**: each wound crop is classified on its own. EfficientNet-B0 is best, with macro-F1 0.797 [0.707, 0.854] on 213 test crops; the other models score 0.69–0.77, and macro AUC is 0.947–0.972.
- **Limitations**: internal validation at a single hospital, a small test set, and manual boxes. External validation is needed.

---

## Repository layout

```
data/            frozen split, CV folds, ROI manifests, audits (no images)
  splits/          case-level split and 5-fold CV folds
  roi/             union-crop manifest (Experiment 1)
  roi_box/         per-box crop manifest (Experiment 2)
src/
  data/            metadata validation, case split, CV folds, ROI cropping, audit
  training/        shared training library, CNN-SVM, ensemble, LR selection, CV plan, per-box photo weighting
  evaluation/      metrics, summaries, paired bootstrap, Grad-CAM, verification
scripts/
  slurm/           SLURM job templates
  hpc/             sync, submit, watch and fetch helpers for the Raptor cluster
tests/           pytest suite on synthetic data (CPU, no patient images)
specs/           phase specifications (as built)
tickets/         one ticket per task: decisions, dates, HPC runs and results
docs/            methodology and pipeline diagrams (open docs/diagrams/*.html in a browser)
runbook.md       every command, path and SLURM setting
ROADMAP.md       scope, ground rules and ticket checklist
```

## Getting started

**Local (code and tests only)**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

The tests use synthetic images and run on CPU without pretrained weights or patient data.

**On the HPC (training and evaluation)**

- Real runs use SLURM on MedCMU-HPC Raptor, in the container `/common/sif/pytorch-24.11-py3.sif` (torch 2.6, CUDA 12.6) on H100 GPUs.
- Use `bash scripts/hpc/sync_to_raptor.sh` to upload the committed code. It refuses uncommitted changes, so every run records its exact commit.
- Use `bash scripts/hpc/fetch_results.sh <run>` to bring back aggregate results only.

For example, to submit all scenarios of Experiment 1 and watch them:

```bash
bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_all.sh)
```

[runbook.md](runbook.md) has every step: cropping, training, evaluation, summaries, cross-validation, Grad-CAM and verification, for all three experiments.

## Reproducibility

- **Fixed seed**: seed 42 is used for the split, CV folds, training and bootstrap. The AlexNet ensemble members use seeds 43–45.
- **Run records**: every run saves its full configuration, the git commit, and the library versions in `run_config.json`.
- **Verification**: the scripts re-derive every model selection and reported number from the run files and check that the test set was not used for selection.
  - Experiment 1 (`src/evaluation/verify.py`): 313 checks passed, 0 failed.
  - Experiment 2 (`src/evaluation/verify_box.py`): 203 checks passed, 0 failed.
  - Experiment 3 (`src/evaluation/verify_crop.py`): 157 checks passed, 0 failed.

## Data privacy

Patient photographs, crops, Grad-CAM overlays, model checkpoints and per-image prediction files stay on the HPC and are never committed (see `.gitignore`). The repository holds code, pseudonymised manifests, and aggregate metrics only. Use any patient image in publications only as the ethics approval allows.
