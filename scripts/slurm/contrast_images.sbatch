#!/bin/bash
# T30: CLAHE copy of the raw photographs for the "more contrast" experiment (CPU).
# Submit from the repo root after `source scripts/hpc/contrast_env.sh`:
#   sbatch scripts/slurm/contrast_images.sbatch
# Writes $RAW_IMAGE_ROOT (same tree as raw_data/wound_ml_7classes, enhanced, no EXIF).
#SBATCH --job-name=t30_contrast_images
#SBATCH --account=a260012
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${RAW_IMAGE_ROOT:?source scripts/hpc/contrast_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_ROOT/raw_data/wound_ml_7classes}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "Source: $SOURCE_ROOT -> $RAW_IMAGE_ROOT"

module purge
module load apptainer/1.3.5

run() {
  apptainer exec --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" "$SIF" "$@"
}

run python -m pytest -q tests/test_contrast_dataset.py
run python src/data/create_contrast_dataset.py --image-root "$SOURCE_ROOT" --output-dir "$RAW_IMAGE_ROOT" \
  --workers "$SLURM_CPUS_PER_TASK"

echo "Done at $(date -Is)"
