#!/bin/bash
# T35: pixel-based quality of the T23 test crops against correct per-crop predictions (CPU).
# Submit from the repo root after `source scripts/hpc/relabel_env.sh`:
#   sbatch scripts/slurm/quality_accuracy.sbatch
# Writes $EXPERIMENTS_DIR/quality_accuracy/ (per-crop table, correlations, quintiles, figures).
#SBATCH --job-name=t35_quality_accuracy
#SBATCH --account=a260012
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${EXPERIMENTS_DIR:?source scripts/hpc/relabel_env.sh first}"
: "${DERIVED_DIR:?source scripts/hpc/relabel_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENTS_DIR/quality_accuracy}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"

module purge
module load apptainer/1.3.5

run() {
  apptainer exec --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" --env "WOUND_DATA_DIR=$WOUND_DATA_DIR,MPLCONFIGDIR=/tmp" "$SIF" "$@"
}

run python -m pytest -q tests/test_quality_accuracy.py
run python -W ignore src/evaluation/quality_accuracy.py \
  --predictions-root "$EXPERIMENTS_DIR/roi_crop/crop" \
  --image-root "$DERIVED_DIR/roi_box_p008" \
  --output-dir "$OUTPUT_DIR"

echo "Done at $(date -Is)"
