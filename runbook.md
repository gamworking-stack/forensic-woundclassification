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

## Diagrams

`docs/diagrams/` holds five Archify diagrams of this pipeline. Open the `.html` files in a browser; each `.json` beside it is the source. Every node links to the code lines it describes at the commit pinned in `meta.repository.revision`.

| Diagram | Shows |
|---|---|
| `raptor-pipeline` | Architecture: laptop, Raptor, SLURM jobs and project storage |
| `data-pipeline` | Data flow: metadata → split → ROI crops → training → evaluation |
| `submit-all` | Workflow: how `submit_all.sh` skips, submits and chains the AlexNet ensemble |
| `training-run` | Sequence: one `train.py` run (resume, epoch loop, checkpoints, predictions) |
| `scenario-status` | Lifecycle: a scenario from not submitted to evaluated, with timeout recovery |

After a code change, update the affected `sources` lines and `revision` in the `.json`, then rebuild (on the laptop; install once with `npx skills add tt-a1i/archify -g`):
```bash
node ~/.agents/skills/archify/bin/archify.mjs finalize <architecture|dataflow|workflow|sequence|lifecycle> \
  docs/diagrams/<name>.json docs/diagrams/<name>.html --repo-root . --quality showcase --out-dir .archify/<name>
```
Check receipts go to the git-ignored `.archify/`.
