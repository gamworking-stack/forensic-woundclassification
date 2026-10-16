# Runbook

## MedCMU-HPC Raptor

- Login: `ssh preenapun.s@raptor.med.cmu.ac.th`
- Account `a260012`, project root `/project/a260012_FORWCL`
- Container `/common/sif/pytorch-24.11-py3.sif` via `module load apptainer/1.3.5` (Python 3.12, torch 2.6 NGC 24.11, CUDA 12.6); no micromamba environment is needed
- Partitions: `gpu` (2 nodes x 2 H100 PCIe), `compute` (CPU), `short` (8 h, tiny tests)
- Support: nattawet.sri@cmu.ac.th, weekdays 9:00-16:30

### Connecting

- Raptor (`10.128.1.10`) is only reachable from the CMU network: on campus, or through the university VPN. Check from the laptop: `nc -z -G 8 raptor.med.cmu.ac.th 22 && echo reachable`.
- SSH key: the **laptop's** public key must be in `~/.ssh/authorized_keys` on Raptor. Run `ssh-copy-id preenapun.s@raptor.med.cmu.ac.th` **on the laptop** (running it on Raptor only installs Raptor's own key). Without a key every `ssh`/`rsync` asks for the password.
- Long sessions: `tmux new -s <name>`, detach with Ctrl+B then D, reattach with `tmux attach -t <name>`.

### Data locations (Raptor)

| Path under `/project/a260012_FORWCL` | Contents | Written by |
|---|---|---|
| `raw_data/wound_ml_7classes/` | 1,112 raw photographs by raw label (5.7 GB) | uploaded once, read only |
| `annotations/cvat_yolo_2026_09_30/` | the 7 CVAT YOLO exports | `sync_to_raptor.sh` |
| `derived_data/roi_union_p008/` | ROI crops by split and class | T03 job |
| `torch_cache/` | ImageNet weights (`TORCH_HOME`) | login-node download (T05) |
| `experiments/<run>/<full\|roi>/<family>/` | checkpoints, prediction files, metrics, figures | training, T08, T15 |

Patient images, crops, checkpoints and per-image prediction files stay on Raptor; only manifests and reports are committed.

### Code to Raptor, results back

```bash
bash scripts/hpc/sync_to_raptor.sh     # laptop: upload the committed repo (+ CVAT exports); refuses uncommitted changes
bash scripts/hpc/fetch_results.sh      # laptop: aggregate results of a run into ../results_figures/<run>/
```
`fetch_results.sh` copies metrics, reports, confusion matrices, ROC curves, tuning and ensemble tables, training histories, run configs and `summary/`; it never copies per-image prediction files, checkpoints or images. Sync only committed code so every run's `run_config.json` names the exact commit (`git_dirty: false`).

### Checking progress and failures

```bash
squeue -u $USER                                       # queued (PD) and running (R) jobs
sacct -S today -X --format=JobID,JobName%40,State,ExitCode,Elapsed | grep -v COMPLETED
less logs/<jobname>_<jobid>.err                       # why a job failed
scancel <jobid>                                       # stop a job
```
A scenario is finished when `experiments/<run>/<path>/<family>/test_predictions.csv` exists. After a `TIMEOUT` or `FAILED`, rerun `submit_all.sh` (below): it resubmits only what is missing and resumes from `last.pt`.

## T03 live ROI crops

1. From the laptop, upload the code and the 7 CVAT exports (30 Sep 2026):
   ```bash
   bash scripts/hpc/sync_to_raptor.sh
   ```
   Repo goes to `~/forensic-woundclassification`, zips to `/project/a260012_FORWCL/annotations/cvat_yolo_2026_09_30/`.
2. On Raptor, submit from the repo root:
   ```bash
   cd ~/forensic-woundclassification
   sbatch scripts/slurm/t03_roi_crops.sbatch
   squeue -u $USER
   ```
3. Read `logs/t03_roi_crops_<jobid>.out`. The job passes when it ends with
   `ROI comparison PASSED`: 1,100 crops whose boxes match `data/roi/roi_manifest.csv`.

Crops are written to `/project/a260012_FORWCL/derived_data/roi_union_p008/<split>/<class>/` and stay on the HPC.
Override paths with `IMAGE_ROOT`, `EXPORTS_DIR` or `OUTPUT_DIR`, e.g. `sbatch --export=ALL,OUTPUT_DIR=/path ...`.

## T05 training library

Pretrained weights are downloaded once on the login node (compute nodes may have no internet) into the shared cache:
```bash
cd ~/forensic-woundclassification
module load apptainer/1.3.5
export TORCH_HOME=/project/a260012_FORWCL/torch_cache
apptainer exec --bind /project/a260012_FORWCL /common/sif/pytorch-24.11-py3.sif python -c \
  "from src.training.config import FAMILIES; from src.training.models import build_model; [build_model(f) for f in FAMILIES]"
```
Every training job must `export TORCH_HOME=/project/a260012_FORWCL/torch_cache` before `apptainer exec` (Apptainer passes the host environment through).

Tests (synthetic images, CPU, no downloads; about 5 minutes). Every new login needs the `module load`:
```bash
cd ~/forensic-woundclassification
module load apptainer/1.3.5
srun -p compute -c 4 --mem=16G -t 1:00:00 \
  apptainer exec /common/sif/pytorch-24.11-py3.sif python -m pytest -q tests/test_training.py
```

## Training jobs (one family x one input path per job)

`scripts/slurm/train_single.sbatch` runs `src/training/train.py` on one H100 with `--resume`, so resubmitting the same job after a timeout continues from `last.pt`.
Outputs go to `/project/a260012_FORWCL/experiments/<RUN_NAME>/<full|roi>/<family>/`.

T11 smoke test (2 epochs on the real images, full and ROI):
```bash
cd ~/forensic-woundclassification
bash scripts/hpc/submit_t11_smoke.sh efficientnet_b0
squeue -u $USER
tail -f logs/t11_efficientnet_b0_roi_<jobid>.out
```

T11 passed on 2026-10-01: jobs `221591` (full) and `221592` (roi), EfficientNet-B0, 2 epochs on an H100, 779/164/157 images on both paths, about 3-5 s per epoch.

CNN-SVM (S13 full, S14 roi): same job script with `FAMILY=cnn_svm`. Features are cached in `<output>/features/`, so a rerun only refits the SVM.
```bash
for p in full roi; do
  sbatch -J "cnn_svm_$p" --time=04:00:00 --export=ALL,FAMILY=cnn_svm,INPUT_PATH=$p scripts/slurm/train_single.sbatch
done
```

## Watching jobs live

`scripts/hpc/watch_jobs.sh` follows the `.out` and `.err` logs as they are written and stops when the jobs finish (Ctrl+C stops watching only).
```bash
bash scripts/hpc/watch_jobs.sh                 # all your queued and running jobs
bash scripts/hpc/watch_jobs.sh 221593 221594   # specific jobs
# submit and watch in one go
ids=(); for p in full roi; do
  ids+=("$(sbatch --parsable -J "cnn_svm_$p" --time=04:00:00 --export=ALL,FAMILY=cnn_svm,INPUT_PATH=$p scripts/slurm/train_single.sbatch)")
done; bash scripts/hpc/watch_jobs.sh "${ids[@]}"
```

## AlexNet ensemble (S15 full, S16 roi)

Three AlexNet members (seeds 43, 44, 45) on GPU, then a CPU job that soft-votes them once all three succeed (it is cancelled if a member fails).
Outputs: `/project/a260012_FORWCL/experiments/main/<full|roi>/alexnet_ensemble/` with `member_01..03/`.
```bash
cd ~/forensic-woundclassification
bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_alexnet_ensemble.sh full)
bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_alexnet_ensemble.sh roi)
```

## Evaluation (T08)

Scores every finished scenario in a run from its prediction files, checks that all were tested on the same images, and writes `evaluation_index.csv` (missing scenarios are listed as `missing`). Rerun it whenever more scenarios finish.
```bash
cd ~/forensic-woundclassification
module load apptainer/1.3.5
srun -p compute -c 2 --mem=8G -t 0:30:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/evaluate.py --experiments-root /project/a260012_FORWCL/experiments/main
column -s, -t < /project/a260012_FORWCL/experiments/main/evaluation_index.csv | cut -c1-150
```
Per scenario: `<split>_metrics.json` (test includes case-level 95% CIs), `<split>_classification_report.csv`, `<split>_confusion_matrix.csv` and `.png`, `<split>_roc_curves.csv` and `.png` (one-vs-rest per class plus macro average, T13).

## Summary and full vs ROI (T15)

After `evaluate.py` has scored all 16 scenarios, write the summary table, the paired full-vs-ROI comparison (2,000 case resamples shared by all scenarios) and its forest plot:
```bash
cd ~/forensic-woundclassification
module load apptainer/1.3.5
srun -p compute -c 2 --mem=8G -t 1:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/summarize.py --experiments-root /project/a260012_FORWCL/experiments/main
```
Outputs in `/project/a260012_FORWCL/experiments/main/summary/`: `summary_table.csv`, `full_vs_roi.csv`, `full_vs_roi_macro_f1.png`.

## Submit everything (S01-S16)

`scripts/hpc/submit_all.sh [run_name]` submits every scenario that has no `test_predictions.csv` and is not already queued. Unfinished runs resume from `last.pt`, so it is safe to rerun after a timeout or failure.
```bash
cd ~/forensic-woundclassification
bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_all.sh)
```

S01-S12 are tuned on validation (T08b): each scenario gets 3 GPU runs, one per learning rate (`S01_efficientnet_b0_full_lr1e-5`, `..._lr3e-5`, `..._lr1e-4`, with `--skip-test`), then a select job (`S01_efficientnet_b0_full_select`, `scripts/slurm/tune_select.sbatch`) that starts after all 3 succeed. It picks the best validation macro-F1, writes `tuning_summary.csv`, predicts test for that run only and copies its files into the scenario folder:
```
/project/a260012_FORWCL/experiments/main/<full|roi>/<family>/
  lr_1e-5/  lr_3e-5/  lr_1e-4/     grid runs (best.pt, last.pt, validation predictions; test only in the chosen one)
  tuning_summary.csv  tuning.log   choice and its validation scores
  run_config.json  training_history.csv  validation_predictions.csv  test_predictions.csv   from the chosen run
```
A failed grid run cancels its select job; rerunning `submit_all.sh` resubmits only what is missing. To rerun the selection by hand:
```bash
cd ~/forensic-woundclassification
module load apptainer/1.3.5
export TORCH_HOME=/project/a260012_FORWCL/torch_cache
srun -p gpu --gres=gpu:1 -c 4 --mem=32G -t 1:00:00 apptainer exec --nv --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/training/select_lr.py \
  --scenario-dir /project/a260012_FORWCL/experiments/main/roi/efficientnet_b0
```

## Grad-CAM (T14)

One fixed set of 2 test images per class (seed 42), explained by every fine-tuned family (CNN-SVM excluded; AlexNet ensemble: representative member) on both input paths, with the checkpoint behind each reported result. Needs the ImageNet weights cache.
```bash
cd ~/forensic-woundclassification
module load apptainer/1.3.5
export TORCH_HOME=/project/a260012_FORWCL/torch_cache
srun -p compute -c 8 --mem=32G -t 2:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/gradcam.py --experiments-root /project/a260012_FORWCL/experiments/main
```
Outputs in `experiments/main/gradcam/`: `examples.csv`, `gradcam_results.csv` (prediction, correct, `matches_prediction_file`, and for full images `cam_in_box` vs `box_share`) and `<family>_set<1|2>.png`. **The figures are overlays on patient photographs: they stay on Raptor** (`fetch_results.sh` does not copy them); use them only as the ethics approval allows.

Every test photo of chosen classes, full image (main run) next to each of its per-box crops (T17 run), after T17 has finished:
```bash
srun -p compute -c 8 --mem=32G -t 4:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/gradcam.py --device cpu \
  --experiments-root /project/a260012_FORWCL/experiments/roi_box --box-manifest data/roi_box/roi_manifest.csv \
  --main-root /project/a260012_FORWCL/experiments/main \
  --all-test --classes "cut wound" "penetration wound" scratch
```
Outputs in `experiments/roi_box/gradcam_all/` (the T14 and T17 `gradcam/` folders are not touched): `examples.csv` (the photos, `set` numbering them within each class), `gradcam_results.csv` (a `full` row per photo and a `box` row per crop, with the photo prediction and `full_cam_in_box` vs `box_share` for that crop's box in the full image) and `<family>_<class>.png`, one row per crop: full input and Grad-CAM with the crop's box, crop input and Grad-CAM. These stay on Raptor too.

## Cross-validation (T12b, supplementary)

5 case-grouped folds over train+validation (`data/splits/cv_folds.csv`, frozen; rebuild check: `python src/data/create_cv_folds.py`). Each scenario reuses its main-run settings (chosen lr, best epoch as a fixed epoch count, chosen SVM, member seeds), trains on 4 folds and predicts the 5th; the test split is never loaded. Needs the finished main run.
```bash
cd ~/forensic-woundclassification
bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_cv.sh)     # 110 jobs: 100 GPU + 10 CPU combine
# when all are done:
module load apptainer/1.3.5
srun -p compute -c 2 --mem=8G -t 1:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/cv_summary.py --cv-root /project/a260012_FORWCL/experiments/cv
```
Outputs: `experiments/cv/<full|roi>/<family>/fold_<k>/` (ensemble members in `fold_<k>/member_0<j>/`), `oof_predictions.csv` per scenario, and `experiments/cv/summary/` (`cv_table.csv`, `cv_full_vs_roi.csv`, `cv_full_vs_roi_macro_f1.png`). The plan is `experiments/cv/cv_plan.tsv`; rerunning `submit_cv.sh` resubmits only missing jobs. `fetch_results.sh cv` copies the summary to `../results_figures/cv/` on the laptop.

## Per-box ROI experiment (T17, extra)

One crop per annotated box instead of one union crop per photo; S17-S24, in its own run `roi_box` (the main run is not touched). Plan and decisions: `tickets/T17_box_roi.md`. Needs the finished, evaluated main run for the comparison.
```bash
cd ~/forensic-woundclassification
# 1. crops (CPU), checked against the committed data/roi_box/roi_manifest.csv (1,667 crops)
sbatch --export=ALL,CROP_MODE=individual,OUTPUT_DIR=/project/a260012_FORWCL/derived_data/roi_box_p008,EXPECTED_MANIFEST=data/roi_box/roi_manifest.csv \
  scripts/slurm/t03_roi_crops.sbatch
# 2. train S17-S24 (lr grid + select for S17-S22, CNN-SVM, AlexNet ensemble)
bash scripts/hpc/watch_jobs.sh $(PATHS=box bash scripts/hpc/submit_all.sh roi_box)
# 3. crop-level metrics, then photo level and the paired comparison with the main run
module load apptainer/1.3.5
srun -p compute -c 2 --mem=8G -t 2:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/evaluate.py --experiments-root /project/a260012_FORWCL/experiments/roi_box
srun -p compute -c 2 --mem=8G -t 2:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/box_summary.py \
  --box-root /project/a260012_FORWCL/experiments/roi_box --main-root /project/a260012_FORWCL/experiments/main
```
Outputs: `experiments/roi_box/box/<family>/` (crop-level prediction and metric files; `photo/` holds the photo-level ones) and `experiments/roi_box/summary/` (`box_summary_table.csv`, `box_vs_union.csv`, `box_vs_full.csv`, `box_vs_union_macro_f1.png`). `fetch_results.sh roi_box` copies the summary to `../results_figures/roi_box/` on the laptop.

Supplementary analyses of the per-box run (added after its test results; T17 §7):
```bash
# 4. 5-fold CV (S17-S24, 55 jobs), then the photo-level CV summary against T12b
bash scripts/hpc/watch_jobs.sh $(MAIN_RUN=roi_box CV_RUN=cv_box PATHS=box bash scripts/hpc/submit_cv.sh)
srun -p compute -c 2 --mem=8G -t 2:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/cv_box_summary.py \
  --cv-box-root /project/a260012_FORWCL/experiments/cv_box --cv-main-root /project/a260012_FORWCL/experiments/cv
# 5. Grad-CAM on every crop of the main run's 14 example photos (CPU)
srun -p compute -c 4 --mem=16G -t 3:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/gradcam.py --device cpu \
  --experiments-root /project/a260012_FORWCL/experiments/roi_box --box-manifest data/roi_box/roi_manifest.csv \
  --examples /project/a260012_FORWCL/experiments/main/gradcam/examples.csv
# 6. verification of everything above
srun -p compute -c 4 --mem=16G -t 3:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/verify_box.py \
  --box-root /project/a260012_FORWCL/experiments/roi_box --main-root /project/a260012_FORWCL/experiments/main \
  --cv-root /project/a260012_FORWCL/experiments/cv_box --cv-main-root /project/a260012_FORWCL/experiments/cv
```
Outputs: `experiments/cv_box/summary/` (`cv_box_table.csv`, `cv_box_vs_union.csv`, `cv_box_vs_full.csv`, figure), `experiments/roi_box/gradcam/` (overlays on patient photographs: they stay on Raptor) and `experiments/roi_box/verification/verification_report.csv`. `fetch_results.sh cv_box` copies the CV summary.

## Per-crop ROI experiment (T18, extra)

The T17 crops, each trained, selected and scored on its own: no photo weights, no photo level, no comparison. S25-S32 in their own run `roi_crop`. Plan and decisions: `tickets/T18_crop_roi.md`. Needs the T17 crops in `derived_data/roi_box_p008`.
```bash
cd ~/forensic-woundclassification
# 1. train S25-S32 (lr grid + select for S25-S30, CNN-SVM, AlexNet ensemble)
bash scripts/hpc/watch_jobs.sh $(PATHS=crop bash scripts/hpc/submit_all.sh roi_crop)
# 2. crop-level metrics, summary table, verification
module load apptainer/1.3.5
srun -p compute -c 2 --mem=8G -t 2:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/evaluate.py --experiments-root /project/a260012_FORWCL/experiments/roi_crop
srun -p compute -c 2 --mem=8G -t 1:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/crop_summary.py --crop-root /project/a260012_FORWCL/experiments/roi_crop
srun -p compute -c 4 --mem=16G -t 2:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/verify_crop.py \
  --crop-root /project/a260012_FORWCL/experiments/roi_crop --main-root /project/a260012_FORWCL/experiments/main
```
Outputs: `experiments/roi_crop/crop/<family>/` (prediction, metric and figure files), `experiments/roi_crop/summary/crop_summary_table.csv`, `experiments/roi_crop/verification/verification_report.csv`. `fetch_results.sh roi_crop` copies them to `../results_figures/roi_crop/`.

## Auto-box ROI experiment (T19, extra)

Plan and every decision: `tickets/T19_auto_box_roi.md` (decision record `.scratch/auto-box-roi/map.md`). Steps marked **(laptop)** need the CVAT exports; everything else runs on Raptor from `~/forensic-woundclassification` after `module load apptainer/1.3.5`. Every Raptor step is one `scripts/slurm/exp4.sbatch` job: `PY_ARGS` is the script and its arguments, `VENV` the venv (default `venv_exp4`); CPU steps add `-p compute --gres=none`.

**0. Reference boxes (laptop, once; committed):**
```bash
python src/detection/boxes.py --exports ~/Downloads/task_*_annotations_2026_09_30_*"yolo 1.1"*.zip
```
`data/roi_auto/reference_boxes.csv` (1,667 unpadded manual boxes, checked against every per-box crop) and `reference_limits.json` (oversize share 0.443, cap 35, size bins).

**1. Environment (Raptor, once).** `venv_exp4` and `venv_exp4_yolo` exist from the T19 check (`.scratch/auto-box-roi/check_raptor.sh`). Add what is still missing:
```bash
P=/project/a260012_FORWCL; SIF=/common/sif/pytorch-24.11-py3.sif; CA=/etc/ssl/certs/ca-certificates.crt
E="HF_HOME=$P/hf_cache,SSL_CERT_FILE=$CA,REQUESTS_CA_BUNDLE=$CA,PIP_CERT=$CA"
apptainer exec --bind $P --env "$E" $SIF $P/venv_exp4/bin/pip install -q "transformers==4.52.4" picklescan
apptainer exec --bind $P --env "$E" $SIF $P/venv_exp4_yolo/bin/pip install -q picklescan safetensors
```
Shell variables for the steps below: `X=$P/experiments; RAW=$P/raw_data/wound_ml_7classes; J="sbatch --parsable scripts/slurm/exp4.sbatch"` (prefix `-J name --export=ALL,PY_ARGS="..."` before the script as shown).

**2. Arm 4F (fine-tuned Faster R-CNN).**
```bash
for lr in 0.005 0.01 0.02; do
  sbatch -J 4F_lr$lr --time=24:00:00 --export=ALL,PY_ARGS="src/detection/finetune.py train --lr $lr --output-dir $X/auto_4F/detector/lr$lr --image-root $RAW" scripts/slurm/exp4.sbatch
done
# when the three finish: the selection files, then the selection (CPU)
mkdir -p $X/auto_4F/detection/dev
for lr in 0.005 0.01 0.02; do ln -s $X/auto_4F/detector/lr$lr/validation.csv $X/auto_4F/detection/dev/lr$lr.csv; done
sbatch -J 4F_select -p compute --gres=none --export=ALL,PY_ARGS="src/detection/select.py --arm 4F --detections-dir $X/auto_4F/detection/dev --configs lr0.005 lr0.01 lr0.02 --output $X/auto_4F/detection/selection.json" scripts/slurm/exp4.sbatch
```
If the gate passed: the five cross-fitted detectors at the chosen learning rate and its best epoch count, then the test detections.
```bash
C=$(python3 -c "import json;print(json.load(open('$X/auto_4F/detection/selection.json'))['chosen_config'])")
BEST=$(python3 -c "import csv;r=list(csv.DictReader(open('$X/auto_4F/detector/$C/history.csv')));print(max(r,key=lambda x:float(x['validation_ap50']))['epoch'])")
for k in 1 2 3 4 5; do
  sbatch -J 4F_fold$k --time=24:00:00 --export=ALL,PY_ARGS="src/detection/finetune.py train --lr ${C#lr} --holdout-fold $k --epochs $BEST --output-dir $X/auto_4F/detector/crossfit/fold$k --image-root $RAW" scripts/slurm/exp4.sbatch
done
sbatch -J 4F_test --export=ALL,PY_ARGS="src/detection/finetune.py detect --run-dir $X/auto_4F/detector/$C --splits test --selection $X/auto_4F/detection/selection.json --output $X/auto_4F/detection/test/$C.csv --image-root $RAW" scripts/slurm/exp4.sbatch
```
(`python3` here is the login node's, standard library only; `max` keeps the first best epoch, as `idxmax` does.)

**3. Arm 4Z (zero-shot).**
```bash
for m in gdino owlv2; do
  sbatch -J 4Z_$m --time=24:00:00 --export=ALL,PY_ARGS="src/detection/zero_shot.py --model $m --splits train validation --output-dir $X/auto_4Z/detection/dev --image-root $RAW" scripts/slurm/exp4.sbatch
done
CONFIGS="gdino_P1 owlv2_P1 gdino_P1_tiled owlv2_P1_tiled gdino_P2 owlv2_P2 gdino_P2_tiled owlv2_P2_tiled gdino_P3 owlv2_P3 gdino_P3_tiled owlv2_P3_tiled gdino_P4 owlv2_P4 gdino_P4_tiled owlv2_P4_tiled"  # zero_shot.config_order()
sbatch -J 4Z_select -p compute --gres=none --export=ALL,PY_ARGS="src/detection/select.py --arm 4Z --detections-dir $X/auto_4Z/detection/dev --configs $CONFIGS --output $X/auto_4Z/detection/selection.json" scripts/slurm/exp4.sbatch
# if passed: C=<chosen_config>
sbatch -J 4Z_test --export=ALL,PY_ARGS="src/detection/zero_shot.py --model ${C%%_*} --splits test --config $C --selection $X/auto_4Z/detection/selection.json --output-dir $X/auto_4Z/detection/test --image-root $RAW" scripts/slurm/exp4.sbatch
```

**4. Arm 4P (public YOLOv8).** Download on the login node, then scan, load once and re-save in a job that can write only one folder; copy the re-saved weights into the run:
```bash
REV=0c61ba140e2c800b25e562e092b777082c7a10fe
apptainer exec --bind $P --env "$E" $SIF $P/venv_exp4/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('trapezius60/forensic_wound_detection', revision='$REV', allow_patterns=['best.pt','best_yolov8m_roboflow_v7.pt','best_yolov8x_robov9.pt']))"
SNAP=$P/hf_cache/hub/models--trapezius60--forensic_wound_detection/snapshots/$REV
sbatch -J 4P_prepare -p compute --gres=none --export=ALL,VENV=venv_exp4_yolo,READ_ONLY=1,WRITABLE=$P/scratch_4P,PY_ARGS="src/detection/public.py prepare --weights-dir $P/scratch_4P/weights --snapshot-dir $SNAP" scripts/slurm/exp4.sbatch
mkdir -p $X/auto_4P && cp -r $P/scratch_4P/weights $X/auto_4P/weights
sbatch -J 4P_dev --export=ALL,VENV=venv_exp4_yolo,PY_ARGS="src/detection/public.py detect --weights-dir $X/auto_4P/weights --splits train validation --output-dir $X/auto_4P/detection/dev --image-root $RAW" scripts/slurm/exp4.sbatch
```
WILLIE MINI gets its one load test (clone `github.com/Qian-Group-HRI/Willie` and download `mini/willie_mini_fold0_best.pt` from `QianGroup/willie-weights` on the login node first): `PY_ARGS="src/detection/public.py willie-smoke --repo <clone> --weights <file> --output $X/auto_4P/willie_smoke.json"`. If it reports `loaded: false`, WILLIE is out (T19 §2.2). If it loads, stop: its inference adapter must be written before it can join the selection. Then:
```bash
sbatch -J 4P_select -p compute --gres=none --export=ALL,PY_ARGS="src/detection/select.py --arm 4P --detections-dir $X/auto_4P/detection/dev --configs v8n_colour v8n_grey v8m_colour v8m_grey v8x_colour v8x_grey --output $X/auto_4P/detection/selection.json" scripts/slurm/exp4.sbatch
# if passed: C=<chosen_config>
sbatch -J 4P_test --export=ALL,VENV=venv_exp4_yolo,PY_ARGS="src/detection/public.py detect --weights-dir $X/auto_4P/weights --splits test --config $C --selection $X/auto_4P/detection/selection.json --output-dir $X/auto_4P/detection/test --image-root $RAW" scripts/slurm/exp4.sbatch
```

**5. Per passing arm (`A`, its chosen config `C`): detection metrics and auto crops (CPU).**
```bash
sbatch -J ${A}_detmetrics -p compute --gres=none --time=6:00:00 --export=ALL,PY_ARGS="src/detection/evaluate_detection.py --detections $X/auto_$A/detection/test/$C.csv --selection $X/auto_$A/detection/selection.json --split test --output-dir $X/auto_$A/detection/test_metrics" scripts/slurm/exp4.sbatch
# crops: 4Z/4P use dev/$C.csv + test/$C.csv; 4F uses the five cross-fitted files + dev/$C.csv + test/$C.csv
DET="$X/auto_$A/detection/dev/$C.csv $X/auto_$A/detection/test/$C.csv"
[[ $A == 4F ]] && DET="$X/auto_4F/detector/crossfit/fold*/train_fold*.csv $DET"
sbatch -J ${A}_crops -p compute --gres=none --export=ALL,PY_ARGS="src/data/create_auto_roi_dataset.py --selection $X/auto_$A/detection/selection.json --detections $DET --output-dir data/roi_auto/$A --image-root $RAW --crop-root $P/derived_data/roi_auto_$A" scripts/slurm/exp4.sbatch
```
The manifests land in the repo's `data/roi_auto/<arm>/{union,box}/`; commit them from Raptor or copy them back to the laptop.

**6. Primary analysis: existing models on the auto crops (GPU, bf16 as in the reported runs).**
```bash
sbatch -J ${A}_primary_union --export=ALL,PY_ARGS="src/evaluation/predict_on_manifest.py --experiments-root $X/main --design union --manifest data/roi_auto/$A/union/roi_manifest.csv --image-root $P/derived_data/roi_auto_${A}_union --output-root $X/auto_$A/primary --check-manual" scripts/slurm/exp4.sbatch
sbatch -J ${A}_primary_box --export=ALL,PY_ARGS="src/evaluation/predict_on_manifest.py --experiments-root $X/roi_box --design box --manifest data/roi_auto/$A/box/roi_manifest.csv --image-root $P/derived_data/roi_auto_${A}_box --output-root $X/auto_$A/primary --check-manual" scripts/slurm/exp4.sbatch
```
`check_manual_<design>.json` must show the same predicted class for (nearly) every manual crop.

**7. Retraining (order 4F → 4Z → 4P; S33-S104):**
```bash
AUTO_ARM=$A PATHS=roi  bash scripts/hpc/watch_jobs.sh $(AUTO_ARM=$A PATHS=roi  bash scripts/hpc/submit_all.sh auto_${A}_union)
AUTO_ARM=$A PATHS=box  bash scripts/hpc/watch_jobs.sh $(AUTO_ARM=$A PATHS=box  bash scripts/hpc/submit_all.sh auto_${A}_box)
AUTO_ARM=$A PATHS=crop bash scripts/hpc/watch_jobs.sh $(AUTO_ARM=$A PATHS=crop bash scripts/hpc/submit_all.sh auto_${A}_crop)
# then, for each of the three roots (CPU):  src/evaluation/evaluate.py --experiments-root $X/auto_${A}_<union|box|crop> --arm $A
```

**8. Comparisons, Grad-CAM and verification (after every passing arm's steps 5-7):**
```bash
sbatch -J auto_compare -p compute --gres=none --time=12:00:00 --export=ALL,PY_ARGS="src/evaluation/auto_vs_manual.py --experiments-dir $X --output-dir $X/auto_summary" scripts/slurm/exp4.sbatch
sbatch -J ${A}_gradcam -p compute --gres=none --export=ALL,PY_ARGS="src/evaluation/gradcam.py --device cpu --experiments-root $X/main --auto-arm $A --auto-root $P/derived_data/roi_auto_${A}_union --auto-manifest data/roi_auto/$A/union/roi_manifest.csv --examples $X/main/gradcam/examples.csv" scripts/slurm/exp4.sbatch
sbatch -J ${A}_gradcam_box -p compute --gres=none --export=ALL,PY_ARGS="src/evaluation/gradcam.py --device cpu --experiments-root $X/roi_box --auto-arm $A --auto-root $P/derived_data/roi_auto_${A}_box --box-manifest data/roi_auto/$A/box/roi_manifest.csv --examples $X/main/gradcam/examples.csv" scripts/slurm/exp4.sbatch
sbatch -J auto_verify -p compute --gres=none --export=ALL,PY_ARGS="src/evaluation/verify_auto.py --experiments-dir $X" scripts/slurm/exp4.sbatch
```
Outputs: `experiments/auto_summary/` (`primary_noninferiority.csv`, `primary_by_family.csv`, `secondary_*.csv`, `arms.json`, `verification/verification_report.csv`), `auto_<arm>/detection/test_metrics/`, and Grad-CAM overlays in `main|roi_box/gradcam_auto_<arm>/` (patient photographs: they stay on Raptor).

## Final verification (T16)

After the main run, T08, T15, T14 and the CV summary: re-derive every selection and reported number from the run files (about 10 minutes on a compute node; the bootstrap intervals are recomputed with the same seed):
```bash
cd ~/forensic-woundclassification
module load apptainer/1.3.5
srun -p compute -c 4 --mem=16G -t 3:00:00 apptainer exec --bind /project/a260012_FORWCL \
  /common/sif/pytorch-24.11-py3.sif python src/evaluation/verify.py \
  --experiments-root /project/a260012_FORWCL/experiments/main --cv-root /project/a260012_FORWCL/experiments/cv
```
It prints PASS/FAIL/NOTE counts per check (files, test set, selection, budget, numbers, gradcam, cv), lists every non-PASS row, and exits 1 on any FAIL. The report is `experiments/main/verification/verification_report.csv` (no image paths); `fetch_results.sh` copies it. What each check does: `src/evaluation/verify.py` docstring and `tickets/T16_final_verification.md`.

## Diagrams

`docs/diagrams/` holds eight Archify diagrams of this pipeline. Open the `.html` files in a browser; each `.json` beside it is the source. Every node links to the code lines it describes at the commit pinned in `meta.repository.revision`.

| Diagram | Shows |
|---|---|
| `raptor-pipeline` | Architecture: laptop, Raptor, SLURM jobs and project storage |
| `data-pipeline` | Data flow: metadata → split → ROI crops → training → evaluation |
| `submit-all` | Workflow: how `submit_all.sh` skips, submits and chains the AlexNet ensemble |
| `training-run` | Sequence: one `train.py` run (resume, epoch loop, checkpoints, predictions) |
| `scenario-status` | Lifecycle: a scenario from not submitted to evaluated, with timeout recovery |
| `lr-selection` | Sequence: the T08b learning-rate grid, the choice on validation and the one test pass |
| `predictions-to-numbers` | Data flow: prediction files → `evaluate.py` / `summarize.py` / `cv_summary.py` → tables, checked by `verify.py` |
| `test-access` | Architecture: which code may read the test split (only after every choice) and which never does |

After a code change, update the affected `sources` lines and `revision` in the `.json`, then rebuild (on the laptop; install once with `npx skills add tt-a1i/archify -g`):
```bash
node ~/.agents/skills/archify/bin/archify.mjs finalize <architecture|dataflow|workflow|sequence|lifecycle> \
  docs/diagrams/<name>.json docs/diagrams/<name>.html --repo-root . --quality showcase --out-dir .archify/<name>
```
Check receipts go to the git-ignored `.archify/`.
