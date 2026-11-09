#!/bin/bash
# T34: class-conditional StyleGAN2-ADA on the training wound crops (all 7 classes, 256 px), one H100.
# Submit from the repo root after `source scripts/hpc/relabel_env.sh`:
#   sbatch scripts/slurm/stylegan_train.sbatch
#   sbatch --export=ALL,KIMG=8,SNAP=1,METRICS=none,RUN=smoke scripts/slurm/stylegan_train.sbatch   # smoke test
# Optional: FOLD (a CV fold, T12b: the generator then sees that fold's training folds only), KIMG
# (default 5000), SNAP (default 50 ticks), GAMMA (default 1), METRICS (default kid50k_full), RESUME (a .pkl).
# Code: NVlabs/stylegan3 (--cfg=stylegan2) cloned to $PROJECT_ROOT/tools/stylegan3 (NVIDIA Source Code License,
# non-commercial research). Writes $DERIVED_DIR/generator_crops_256[_fold<k>]/ (dataset) and
# $EXPERIMENTS_DIR/generators/stylegan2/<run>/ (patient-derived: they stay on Raptor).
#SBATCH --job-name=t34_stylegan
#SBATCH --account=a260012
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${DERIVED_DIR:?source scripts/hpc/relabel_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
SG="${SG:-$PROJECT_ROOT/tools/stylegan3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SUFFIX="${FOLD:+_fold$FOLD}"
DATA="$DERIVED_DIR/generator_crops_256$SUFFIX"
RUN="${RUN:-main$SUFFIX}"
OUT="$EXPERIMENTS_DIR/generators/stylegan2/$RUN"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git')); stylegan3 $(git -C "$SG" rev-parse --short HEAD)"

module purge
module load apptainer/1.3.5

run() {
  apptainer exec --nv --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" \
    --env "TORCH_HOME=$PROJECT_ROOT/torch_cache,TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/torch_cache/extensions,DNNLIB_CACHE_DIR=$PROJECT_ROOT/torch_cache/dnnlib" \
    "$SIF" "$@"
}

run python -m pytest -q tests/test_generator_crops.py
if [[ ! -f "$DATA/dataset.json" ]]; then
  run python src/data/export_generator_crops.py --image-root "$DERIVED_DIR/roi_box_p008" --output-dir "$DATA" \
    ${FOLD:+--cv-folds "$WOUND_DATA_DIR/splits/cv_folds.csv" --fold "$FOLD"}
fi
mkdir -p "$OUT"
run python "$SG/train.py" --outdir "$OUT" --cfg stylegan2 --data "$DATA" --gpus 1 --batch 32 \
  --gamma "${GAMMA:-1}" --mirror 1 --cond 1 --kimg "${KIMG:-5000}" --snap "${SNAP:-50}" \
  --metrics "${METRICS:-kid50k_full}" --workers 4 ${RESUME:+--resume "$RESUME"}

echo "Done at $(date -Is)"
