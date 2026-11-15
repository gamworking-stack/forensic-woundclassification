#!/bin/bash
# T37: full-image / crop soft voting with crops from detector boxes (4Y YOLO, 4F Faster R-CNN), photo level, test.
# Needs the per-crop predictions of the auto crops in $OUT/source_<arm>/roi_crop/crop/<family>/ (predict_on_manifest.py
# --design crop) and the T36 outputs in $EXPERIMENTS_DIR/fusion/. Submit from the repo root after
# `source scripts/hpc/relabel_env.sh`:  sbatch scripts/slurm/t37_fusion.sbatch
# Writes $EXPERIMENTS_DIR/fusion_auto/: fused_<arm>/{equal,crop_only}/, patient/<arm>_<rule>/ (T25) and compare/:
#   <arm>_fused_vs_full     vote with auto crops  - full image alone
#   <arm>_fused_vs_manual   vote with auto crops  - vote with manual crops (T36 equal)
#   <arm>_crops_vs_full     auto crops alone      - full image alone
#SBATCH --job-name=t37_fusion
#SBATCH --account=a260012
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${EXPERIMENTS_DIR:?source scripts/hpc/relabel_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${OUTPUT_DIR:-$EXPERIMENTS_DIR/fusion_auto}"
T36="$EXPERIMENTS_DIR/fusion"
ARMS="${ARMS:-4Y 4F}"
FAMILIES="efficientnet_b0 resnet152 mobilenet_v3_small googlenet vgg16 vit_b16 cnn_svm alexnet_ensemble"
declare -A MANIFEST=([4Y]="$EXPERIMENTS_DIR/auto_4Y/roi/box/roi_manifest.csv"
                     [4F]="$WOUND_DATA_DIR/roi_auto/4F/box/roi_manifest.csv")

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
module purge
module load apptainer/1.3.5

run() {
  apptainer exec --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" --env "WOUND_DATA_DIR=$WOUND_DATA_DIR,MPLCONFIGDIR=/tmp" "$SIF" "$@"
}

run python -m pytest -q tests/test_fusion.py tests/test_contrast_compare.py

for arm in $ARMS; do
  SRC="$OUT/source_$arm"
  # The T23 full-image test predictions next to the auto crops' predictions
  for family in $FAMILIES; do
    mkdir -p "$SRC/main/full/$family"
    cp "$T36/original/main/full/$family/test_predictions.csv" "$SRC/main/full/$family/"
  done
  run python -W ignore src/evaluation/fusion.py --source-root "$SRC" --output-root "$OUT/fused_$arm" \
    --box-manifest "${MANIFEST[$arm]}" --splits test --rules equal crop_only --levels photo
  for rule in equal crop_only; do
    run python src/evaluation/patient_rescore.py --source-root "$OUT/fused_$arm/$rule" --output-root "$OUT/patient/${arm}_$rule"
  done
  # Paired patient bootstrap on the same 149 test photographs (T30 method), photo level only
  run python -W ignore src/evaluation/contrast_compare.py --paths full --original-root "$T36/patient/original" \
    --contrast-root "$OUT/patient/${arm}_equal" --output-dir "$OUT/compare/${arm}_fused_vs_full" --name "${arm}_fused"
  run python -W ignore src/evaluation/contrast_compare.py --paths full --original-root "$T36/patient/equal" \
    --contrast-root "$OUT/patient/${arm}_equal" --output-dir "$OUT/compare/${arm}_fused_vs_manual" --name "${arm}_fused"
  run python -W ignore src/evaluation/contrast_compare.py --paths full --original-root "$T36/patient/original" \
    --contrast-root "$OUT/patient/${arm}_crop_only" --output-dir "$OUT/compare/${arm}_crops_vs_full" --name "${arm}_crops"
done

echo "Done at $(date -Is)"
