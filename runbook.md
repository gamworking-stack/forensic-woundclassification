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
