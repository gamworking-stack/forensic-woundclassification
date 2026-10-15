#!/bin/bash
# Train one model family (or FAMILY=cnn_svm) on one input path on one H100.
# Submit from the repo root:
#   sbatch -J S02_efficientnet_b0_roi --export=ALL,FAMILY=efficientnet_b0,INPUT_PATH=roi \
#     scripts/slurm/train_single.sbatch
# Optional: RUN_NAME (default main), EXTRA_ARGS (passed to train.py), OUTPUT_DIR.
# INPUT_PATH=box (T17) trains on one crop per annotated box; use RUN_NAME=roi_box.
# INPUT_PATH=crop (T18) trains on the same crops, each its own sample; use RUN_NAME=roi_crop.
#SBATCH --job-name=train_single
#SBATCH --account=a260012
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${FAMILY:?set FAMILY, e.g. --export=ALL,FAMILY=efficientnet_b0,INPUT_PATH=roi}"
: "${INPUT_PATH:?set INPUT_PATH to full, roi, box or crop}"
RUN_NAME="${RUN_NAME:-main}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

case "$INPUT_PATH" in
  full)
    MANIFEST=data/splits/case_level_split_manifest.csv
    IMAGE_ROOT="$PROJECT_ROOT/raw_data/wound_ml_7classes" ;;
  roi)
    MANIFEST=data/roi/roi_manifest.csv
    IMAGE_ROOT="$PROJECT_ROOT/derived_data/roi_union_p008" ;;
  box|crop)
    MANIFEST=data/roi_box/roi_manifest.csv
    IMAGE_ROOT="$PROJECT_ROOT/derived_data/roi_box_p008" ;;
  *)
    echo "INPUT_PATH must be full, roi, box or crop, got: $INPUT_PATH" >&2; exit 1 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/experiments/$RUN_NAME/$INPUT_PATH/$FAMILY}"
export TORCH_HOME="$PROJECT_ROOT/torch_cache"
export PYTHONUNBUFFERED=1  # log lines reach the .out/.err files immediately

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Family: $FAMILY | Input: $INPUT_PATH | Run: $RUN_NAME | Extra: ${EXTRA_ARGS:-none}"
echo "Output: $OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

module purge
module load apptainer/1.3.5

COMMON_ARGS=(
  --input-path "$INPUT_PATH"
  --manifest "$MANIFEST"
  --image-root "$IMAGE_ROOT"
  --output-dir "$OUTPUT_DIR"
  --num-workers $(( SLURM_CPUS_PER_TASK - 2 ))
)
if [[ "$FAMILY" == cnn_svm ]]; then
  ENTRY=(src/training/cnn_svm.py)
else
  ENTRY=(src/training/train.py --family "$FAMILY" --resume)
fi

# EXTRA_ARGS is split into words on purpose.
# shellcheck disable=SC2086
apptainer exec --nv --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" "$SIF" \
  python "${ENTRY[@]}" "${COMMON_ARGS[@]}" $EXTRA_ARGS

echo "Done at $(date -Is)"
