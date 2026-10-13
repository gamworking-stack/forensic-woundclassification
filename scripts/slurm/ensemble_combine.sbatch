#!/bin/bash
# Soft-vote the finished members in ENSEMBLE_DIR/member_01..03 (CPU only).
# Submitted by scripts/hpc/submit_alexnet_ensemble.sh after all members succeed.
#SBATCH --job-name=ensemble_combine
#SBATCH --account=a260012
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${ENSEMBLE_DIR:?set ENSEMBLE_DIR, e.g. --export=ALL,ENSEMBLE_DIR=/project/.../alexnet_ensemble}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Ensemble: $ENSEMBLE_DIR"

module purge
module load apptainer/1.3.5

apptainer exec --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" "$SIF" \
  python src/training/ensemble.py \
    --members "$ENSEMBLE_DIR"/member_01 "$ENSEMBLE_DIR"/member_02 "$ENSEMBLE_DIR"/member_03 \
    --output-dir "$ENSEMBLE_DIR" \
    --splits ${ENSEMBLE_SPLITS:-validation test}

echo "Done at $(date -Is)"
