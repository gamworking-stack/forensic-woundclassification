#!/bin/bash
# T08b: choose a scenario's learning rate on validation, then predict test for that run (one H100).
# Submitted by scripts/hpc/submit_all.sh after the scenario's grid runs succeed.
#SBATCH --job-name=tune_select
#SBATCH --account=a260012
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${SCENARIO_DIR:?set SCENARIO_DIR, e.g. --export=ALL,SCENARIO_DIR=/project/.../roi/efficientnet_b0}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export TORCH_HOME="$PROJECT_ROOT/torch_cache"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Scenario: $SCENARIO_DIR"

module purge
module load apptainer/1.3.5

apptainer exec --nv --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" "$SIF" \
  python src/training/select_lr.py --scenario-dir "$SCENARIO_DIR" --num-workers $(( SLURM_CPUS_PER_TASK - 2 ))

echo "Done at $(date -Is)"
