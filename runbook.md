# Runbook

## MedCMU-HPC Raptor

- Login: `ssh preenapun.s@raptor.med.cmu.ac.th` (set up a key with `ssh-copy-id` after the first password login)
- Account `a260012`, project root `/project/a260012_FORWCL`
- Container `/common/sif/pytorch-24.11-py3.sif` via `module load apptainer/1.3.5`

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
