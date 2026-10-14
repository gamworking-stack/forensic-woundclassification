# ROADMAP.md: Forensic Wound Classification (project starter)

> Starter roadmap for Claude Code running in Antigravity. Read this file at the start of every session. It defines scope, order of work, and what "done" means. The repo starts empty; everything below is built from scratch.

## 0. How to work

1. **Start each session with the `wayfinder` skill** to re-orient: read this file and `runbook.md`, check git status, and pick the next open ticket.
2. **`/to-spec`**: turn the next phase into a short spec (inputs, outputs, files, acceptance checks) in `specs/<phase>.md`.
3. **`/to-ticket`**: break the spec into small tickets in `tickets/` (one ticket = one script or one job, plus its check).
4. **`/implement`**: build one ticket, run its check, tick it off in section 9, record its files used, created and changed in `tickets/TICKET_MEMO.md`, commit.
5. Update `runbook.md` whenever a command, path, or SLURM setting changes.

Ground rules:

- **One fixed case-level split**, created once and never regenerated. Every experiment reads it.
- **No leakage:** all images with the same `pseudonimized_id` live in exactly one split. Every data script asserts this.
- **Test set is used once per scenario**, after model selection on validation. Never tune on test.
- **Local vs HPC:** Antigravity is for writing, linting, dry runs, and tiny smoke tests. Real training runs only on the HPC through SLURM.
- **Reproducibility:** fixed seed (42), config saved with every run, same training budget for every scenario.

## 1. Goal

Classify 7 wound types from forensic images: cut wound, graze wound, gunshot wound, contusion, laceration, scratch, penetration wound.

Compare **8 model families x 2 input paths (full image, ROI) = 16 scenarios** on one fixed test set.

Source of truth: the thesis proposal (Preenapun Saokhieo, 690631105). If this roadmap and the proposal disagree, the proposal wins; flag the conflict to the user.

Study framing from the proposal: ROI-assisted classification is the main idea. The central question is **whether wound localization (bounding-box ROI crops) improves classification over full images**. Full-image models are the comparison baseline. The study is internal validation only (single hospital, no external test set), and the main task is image-level classification.

## 2. Inputs

| Item | Detail |
|---|---|
| Images | Forensic wound photographs, 2021 to 2026, university-affiliated hospital in Thailand. Total and per-class counts are **not fixed**; the code reads whatever the metadata CSV contains and reports the counts |
| Metadata | `thesis_data_id.csv` with `pseudonimized_id`, image filename, `wound_label`. The proposal's full CSV also has `wound_id`, `body_site`, `patient_age`, `race_ethnicity`, `skin_tone_mst` (Monk scale, may be unassessable), `autopsy_status` |
| Annotations | CVAT export zip(s) with manual bounding boxes of visible wound areas |
| Compute | MedCMU-HPC "Raptor" cluster (Faculty of Medicine, CMU), SLURM. Details in section 2b |

### 2b. MedCMU-HPC facts (from the cluster's documentation site)

| Item | Detail |
|---|---|
| OS | Rocky Linux 8.10, local SSD on each node |
| GPU nodes | 2 nodes, each 2x AMD EPYC 9454 (96 cores), 384 GB RAM, **2x NVIDIA H100** (4 GPUs total) |
| Partition `gpu` | max job time **7 days**, max **192 cores per user** |
| Partition `compute` | CPU only, 96 cores and 1,280 GB RAM per node, 7 days, 192 cores per user |
| Partition `short` | 8 hours, 2 cores per user, 4 GB RAM per node, CPU only; use for quick CPU tests (no GPU) |
| Access | `ssh <username>@<host>`; SSH keys recommended (`ssh-keygen -t ed25519`, `ssh-copy-id`); MobaXterm on Windows. Web portal (Open OnDemand) at `raptor.med.cmu.ac.th:9000` |
| Jobs | `sbatch script.sbatch`, `srun -p short -t 1:00:00 --mem=4G -c 1 --pty python`, `squeue`, `scancel <id>` |
| Software | Environment modules (`module avail`, `ml spider <name>`, `module load`, `module purge`); Python environments with **micromamba** (`module` for mamba 1.5.8, personal packages in `~/micromamba/pkgs`) |
| Transfer | `rsync -rvz <source> <username>@<host>:<dest>` |
| Long sessions | `tmux new -s <name>`, detach with Ctrl+B then D, reattach with `tmux attach -t <name>` |
| Storage | Scratch (temporary processing) and project directories (long-term). Exact paths and quotas are **not documented**; confirm with support |
| Support | nattawet.sri@cmu.ac.th, weekdays 9:00 to 16:30, Research Office room, 9th floor, 50th Year Anniversary Building; booking at cal.com/nattawet-sri/30min |
| Access requests | User and project registration forms on the cluster site; service fee calculator on the site |

**Still unknown (ask support or check on the cluster):** login hostname, GPU request flag in SLURM (likely `--gres=gpu:N`, verify), SLURM account/QOS if required, module names for CUDA and mamba, storage paths and quotas, per-user GPU limit, whether jobs are billed by GPU-hour.

**Label definitions (from the proposal, DiMaio):** the label is the *dominant* wound type from the case description, confirmed by expert review. Graze = superficial tangential scrape; scratch = narrow linear scrape; firearm-related graze stays **gunshot wound**; penetration = non-firearm pointed-object entry (stab or puncture). Keep these in the runbook so label handling in code matches the thesis.

## 3. Experiment design

**Model families (8):** MobileNetV3-Small, VGG16, ResNet152, EfficientNet-B0, CNN-SVM (pretrained CNN features + SVM), AlexNet ensemble (soft voting), GoogLeNet, ViT-B/16.

**Input paths (2):** full image, and ROI crops cut from the CVAT annotations.

**Metrics per scenario (proposal):** accuracy, balanced accuracy, precision, recall (sensitivity), F1, macro F1, class-wise precision/recall/F1, confusion matrix picture, ROC and AUC curves picture. Grad-CAM for qualitative interpretability "where applicable".

**Fixed by the proposal:**

- Pretrained models fine-tuned for a 7-class output; all models share the same split, preprocessing, full-vs-ROI framework, and metrics.
- Resizing and normalization; augmentation from this list: rotation, flipping, zooming, contrast adjustment, noise addition.
- Split: random, case-level, 70:15:15, all images of a case in one subset.
- Hyperparameter tuning (learning rate, batch size, epochs, dropout) by grid search or Bayesian optimization; k-fold cross-validation if the data size allows.
- CNN-SVM: hybrid of VGG16, ResNet152, EfficientNet features feeding an SVM (ResNet152 named as the key extractor).
- AlexNet ensemble: several fine-tuned AlexNets combined by soft voting.

**Defaults to propose in the first spec (not in the proposal, adjust with the user):** class-weighted loss, early stopping on validation macro F1, pad-to-square before resize, 384 px input (ViT-B/16 at 224 px).

## 4. Repository layout

```
src/            data, models, training, evaluation, visualization
scripts/slurm/  .sbatch templates and submit_all.sh
data/           split + ROI manifests (not committed if sensitive)
results/        full/<family>/  roi/<family>/  summary/
specs/  tickets/  logs/
runbook.md  ROADMAP.md  requirements.txt
```

As built, the model builder is `src/training/models.py`, the figures are drawn in `src/evaluation/plots.py`, and run outputs stay on Raptor (`specs/phase0_environment.md` §2).

## 5. Phases and tickets

### Phase 0: Environment
- **T00** Create the skeleton above and `requirements.txt` (PyTorch, torchvision, scikit-learn, pandas, matplotlib, grad-cam). *Check:* imports succeed in a fresh environment.

### Phase 1: Dataset and split
- **T01** Validate the metadata CSV: required columns, no empty values, 7 expected labels, no duplicate image paths, every file exists. Normalize label spelling variants to the 7 canonical names. *Check:* validation report with zero errors; all 7 classes are present, and the observed total and per-class image and case counts are written to the report (no hard-coded counts anywhere in the code); optionally check inclusion criteria (age over 18, dates 1 Jan 2021 to 30 Apr 2026) if those columns are present.
- **T02** Build the 70/15/15 split grouped by `pseudonimized_id`. A case may contain more than one wound class, so search many random group assignments and keep the one that best preserves class balance, with every class present in every split. *Check:* leakage assertion passes; per-class image and case counts per split saved; the manifest is committed and frozen. *Note for the thesis:* the proposal says "randomly divided"; a seeded random search that enforces class presence is a refinement and should be described in the methods.

### Phase 2: ROI preprocessing
- **T03** Parse the CVAT zip(s) and crop wound regions, with padding (start at 8%) and a minimum crop size. Combine all boxes of an image into one crop so one image gives one ROI sample. Each crop inherits the case ID and split of its source image. Support a dry run. *Check:* crop summary lists skipped annotations (not in manifest, missing image, too small); every class appears in train, validation, and test.
- **T04** Audit full vs ROI: class and case counts per split for both paths. *Check:* table saved; any image lost in the ROI path is documented so the paths can be compared fairly.

### Phase 3: Training code
- **T05** Shared library: dataset loader from a manifest, transforms, model builder for all 8 families, one train/evaluate loop, checkpoint and resume support. *Check:* a 2-epoch smoke test on a tiny subset runs end to end.
- **T06** CNN-SVM: extract frozen features from several pretrained CNNs, tune the SVM on validation, refit on train+validation, predict test.
- **T07** AlexNet ensemble: train several members with different seeds, average their probabilities.
- **T08** Evaluation module: all proposal metrics (including balanced accuracy and class-wise scores) plus saved per-image predictions with class probabilities. *Check:* output files exist for every scenario.
- **T08b** Hyperparameter tuning (learning rate, batch size, epochs, dropout) by grid search or Bayesian optimization on the validation set, using a small shared search space per family so the budget stays comparable. Optional k-fold grouped cross-validation on train+validation only if the data size allows. *Check:* chosen values recorded per scenario; test set untouched.

### Phase 4: HPC
- **T09** SLURM templates for single-model, CNN-SVM, and ensemble jobs, plus `submit_all.sh` that skips finished scenarios and resumes interrupted ones. Use `-p gpu`, one H100 per job (training is single-GPU, so up to 4 scenarios can run in parallel), `--cpus-per-task` set to match `--num-workers` (start with 8 to 12), and a time limit under 7 days with `--resume` so a timeout is not fatal. CPU-only steps (split, ROI crops, SVM tuning, figures) go to `-p compute`; `short` (8 hours, 2 cores, 4 GB) is only for tiny tests. Fill the unknowns in section 2b first. *Check:* `sbatch --test-only` passes.
- **T09b** Environment on the cluster: create a micromamba environment from `requirements.txt` (PyTorch with CUDA matching the installed driver); verify with a short GPU `srun` job that `torch.cuda.is_available()` is true. Run long jobs inside `tmux` or via `sbatch`, never on the login node.
- **T09c** Data transfer: copy the frozen manifest, images, and CVAT zips to the cluster project or scratch directory with `rsync -rvz`; copy `results/` back the same way. Keep patient data on the cluster; never commit it.
- **T10** `runbook.md`: connection (`ssh`, keys), environment, data locations, submit, monitor, resume, where results go, support contact.

### Phase 5: Experiments
- **T11** Smoke-test one family on the HPC with 2 epochs.
- **T12** Run all 16 scenarios (S01 to S16). Order: EfficientNet-B0, ResNet152, MobileNetV3-Small, GoogLeNet, VGG16, ViT-B/16, CNN-SVM, AlexNet ensemble; each on ROI and full image.
- *Optional:* 5-fold grouped cross-validation on development data (train+validation only) for the strongest models, to report mean and SD.

### Phase 6: Figures and results
- **T13** ROC curves (one-vs-rest, 7 classes) with AUC per class and macro AUC, from saved predictions.
- **T14** Grad-CAM for each CNN family and the ViT (needs a transformer-specific reshape), using a fixed rule to pick example images (for example 2 correct and 2 wrong per class) so full and ROI are comparable. The proposal limits Grad-CAM to "where applicable", so CNN-SVM may be excluded and documented as such; confirm with the user how to treat the AlexNet ensemble (for example, one member).
- **T15** One summary table of all 16 scenarios (accuracy, balanced accuracy, precision, recall, macro F1, macro AUC), plus a full-vs-ROI comparison per family on the same test cases (the proposal's main research question). Optional: simple subgroup breakdowns by `autopsy_status` or skin tone if those columns are available.
- **T16** Final verification: 16 of 16 scenarios present, all figures exist, test never used for selection, summary numbers match per-run files.

### Extra experiments (outside the 16-scenario study)
- **T17** Per-box ROI: one crop per annotated box instead of the union crop (S17-S24, run `roi_box`), photo-level primary outcome (mean of crop probabilities), compared with union ROI and full image on the same test cases. Separate manifest, crops and run; the main study is unchanged. Plan: `tickets/T17_box_roi.md`.

## 6. Risks to track

- **Few independent cases** for rare classes make small test sets noisy. Report case counts next to every metric.
- **Class imbalance:** use class weights and macro metrics.
- **ROI vs full comparability:** images without a usable annotation drop out of the ROI path.
- **Compute:** ViT-B/16 and ResNet152 are the heaviest; use checkpoint resume so wall-time limits are not fatal.

## 7. Definition of done

All 16 scenarios finished under identical split, seed, and budget policy; summary table, confusion matrices, ROC/AUC curves, and Grad-CAM images generated; `runbook.md` lets another person reproduce every number.

## 8. Questions for the user (ask in the first `/to-spec`)

1. The unknown cluster details in section 2b (login host, GPU flag, account, storage paths, CUDA/mamba module names, GPU limits). The user may need to email HPC support.
2. Training budget (max epochs, patience) shared by all scenarios.
3. Grad-CAM for the AlexNet ensemble (CNN-SVM may be excluded under "where applicable"); and whether CNN-SVM uses concatenated features from all three CNNs or ResNet152 alone.
4. Is the data too sensitive to store in the repo (manifests only, images stay on HPC)?

## 9. Progress checklist

- [x] T00 skeleton
- [x] T01 metadata validation
- [x] T02 case-level split
- [x] T03 ROI crops
- [x] T04 full vs ROI audit
- [x] T05 shared training library
- [x] T06 CNN-SVM
- [x] T07 AlexNet ensemble
- [x] T08 evaluation module
- [x] T08b hyperparameter tuning (and optional CV)
- [x] T09 SLURM templates
- [x] T09b cluster environment (micromamba, GPU check)
- [x] T09c data transfer with rsync
- [x] T10 runbook.md
- [x] T11 HPC smoke test
- [x] T12 16 scenarios (S01 to S16)
- [x] T13 ROC/AUC curves
- [x] T14 Grad-CAM
- [x] T15 summary table
- [x] T16 final verification
- [x] T17 per-box ROI experiment (extra)
