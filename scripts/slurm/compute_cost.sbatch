#!/bin/bash
# T33: computational cost of the 8 model families (parameters, FLOPs, GPU and CPU latency,
# GPU throughput, peak training memory) on one H100, with the T23 CNN-SVM runs' SVMs refitted.
# Submit from the repo root after `source scripts/hpc/relabel_env.sh`:
#   sbatch --nodelist=raptor-g-002 scripts/slurm/compute_cost.sbatch
# (raptor-g-001 also lends its GPU out in shards, which would disturb the timing.)
# Writes $EXPERIMENTS_DIR/compute_cost/: compute_cost.csv, svm_cost.csv, compute_cost.json.
#SBATCH --job-name=t33_compute_cost
#SBATCH --account=a260012
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${EXPERIMENTS_DIR:?source scripts/hpc/relabel_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENTS_DIR/compute_cost}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"

module purge
module load apptainer/1.3.5

run() {
  apptainer exec --nv --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" --env "TORCH_HOME=$PROJECT_ROOT/torch_cache" "$SIF" "$@"
}

run nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
run python -m pytest -q tests/test_compute_cost.py
run python src/evaluation/compute_cost.py --device cuda --cpu-threads "$SLURM_CPUS_PER_TASK" \
  --svm-runs "$EXPERIMENTS_DIR/main/full/cnn_svm" "$EXPERIMENTS_DIR/roi_crop/crop/cnn_svm" \
  --output-dir "$OUTPUT_DIR"

echo "Done at $(date -Is)"
