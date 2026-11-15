#!/bin/bash
# T36: soft voting between the full-image and crop models of each family, scored against the originals (CPU).
# Submit from the repo root after `source scripts/hpc/relabel_env.sh`:
#   sbatch scripts/slurm/fusion.sbatch
# Writes $EXPERIMENTS_DIR/fusion/: fused/ (equal, tuned, fusion_weights.csv), original/ (copies of the
# prediction files), patient/ (all rescored per patient, T25) and compare_{equal,tuned}/ (T30 comparison).
#SBATCH --job-name=t36_fusion
#SBATCH --account=a260012
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${EXPERIMENTS_DIR:?source scripts/hpc/relabel_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${OUTPUT_DIR:-$EXPERIMENTS_DIR/fusion}"
FAMILIES="efficientnet_b0 resnet152 mobilenet_v3_small googlenet vgg16 vit_b16 cnn_svm alexnet_ensemble"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"

module purge
module load apptainer/1.3.5

run() {
  apptainer exec --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" --env "WOUND_DATA_DIR=$WOUND_DATA_DIR,MPLCONFIGDIR=/tmp" "$SIF" "$@"
}

run python -m pytest -q tests/test_fusion.py tests/test_contrast_compare.py

# 1. Fused prediction files, both rules
run python -W ignore src/evaluation/fusion.py --source-root "$EXPERIMENTS_DIR" --output-root "$OUT/fused"

# 2. The original validation and test files, in the same layout
for family in $FAMILIES; do
  for rp in main/full roi_crop/crop; do
    mkdir -p "$OUT/original/$rp/$family"
    cp "$EXPERIMENTS_DIR/$rp/$family/"{validation,test}_predictions.csv "$OUT/original/$rp/$family/"
  done
done

# 3. One case per patient (T25), originals and fused alike
for name in original fused/equal fused/tuned; do
  run python src/evaluation/patient_rescore.py --source-root "$OUT/$name" --output-root "$OUT/patient/$(basename "$name")"
done

# 4. Fused minus original on the same test images, paired patient bootstrap (T30)
for rule in equal tuned; do
  run python -W ignore src/evaluation/contrast_compare.py --original-root "$OUT/patient/original" \
    --contrast-root "$OUT/patient/$rule" --output-dir "$OUT/compare_$rule" --name "fused_$rule"
done
cp "$OUT/fused/fusion_weights.csv" "$OUT/compare_tuned/"

echo "Done at $(date -Is)"
