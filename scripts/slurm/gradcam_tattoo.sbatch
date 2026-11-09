#!/bin/bash
# T31: Grad-CAM panels of the tattoo photos (train, validation, test) for every model behind the
# manuscript tables: T23 (Tables 2, 2b, 3) and T30 CLAHE (Table 2c), full image and per crop (CPU).
# Submit from the repo root after `source scripts/hpc/relabel_env.sh`, with the list from subgroup_photos.py:
#   sbatch --export=ALL,PHOTO_LIST=/project/a260012_FORWCL/experiments_relabel_2026_10_23/tattoo_photos.csv \
#     scripts/slurm/gradcam_tattoo.sbatch
# Writes <run>/roi_crop/gradcam_tattoo/ in both experiment folders (patient photographs: they stay on Raptor).
#SBATCH --job-name=t31_gradcam_tattoo
#SBATCH --account=a260012
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${PHOTO_LIST:?set PHOTO_LIST to the CSV written by subgroup_photos.py}"
: "${WOUND_DATA_DIR:?source scripts/hpc/relabel_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
RUNS="${RUNS:-$PROJECT_ROOT/experiments_relabel_2026_10_23 $PROJECT_ROOT/experiments_relabel_2026_10_23_more_contrast}"
# CNN-SVM has no Grad-CAM (T14 §2)
FAMILIES="${FAMILIES:-efficientnet_b0 resnet152 mobilenet_v3_small googlenet vgg16 vit_b16 alexnet_ensemble}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"

module purge
module load apptainer/1.3.5

run() {
  apptainer exec --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" --env "TORCH_HOME=$PROJECT_ROOT/torch_cache" "$SIF" "$@"
}

run python -m pytest -q tests/test_gradcam.py tests/test_subgroup.py
for X in $RUNS; do
  echo "== $X"
  # shellcheck disable=SC2086
  run python src/evaluation/gradcam_panels.py --main-root "$X/main" --crop-root "$X/roi_crop" \
    --box-manifest "$WOUND_DATA_DIR/roi_box/roi_manifest.csv" --photo-list "$PHOTO_LIST" \
    --families $FAMILIES --output-dir "$X/roi_crop/gradcam_tattoo" --device cpu
done

echo "Done at $(date -Is)"
