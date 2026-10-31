#!/bin/bash
# Train one model family (or FAMILY=cnn_svm) on one input path on one H100.
# Submit from the repo root:
#   sbatch -J S02_efficientnet_b0_roi --export=ALL,FAMILY=efficientnet_b0,INPUT_PATH=roi \
#     scripts/slurm/train_single.sbatch
# Optional: RUN_NAME (default main), EXTRA_ARGS (passed to train.py), OUTPUT_DIR.
# INPUT_PATH=box (T17) trains on one crop per annotated box; use RUN_NAME=roi_box.
# INPUT_PATH=crop (T18) trains on the same crops, each its own sample; use RUN_NAME=roi_crop.
# AUTO_ARM=4Z|4P|4F (T19) swaps in that arm's auto-box crops; use RUN_NAME=auto_<arm>_<union|box|crop>.
# WOUND_DATA_DIR, EXPERIMENTS_DIR and DERIVED_DIR select another dataset version (T23: source scripts/hpc/relabel_env.sh).
# RAW_IMAGE_ROOT replaces the raw photographs for INPUT_PATH=full (T30: source scripts/hpc/contrast_env.sh).
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
DATA_DIR="${WOUND_DATA_DIR:-data}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$PROJECT_ROOT/experiments}"
DERIVED_DIR="${DERIVED_DIR:-$PROJECT_ROOT/derived_data}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

case "$INPUT_PATH" in
  full)
    MANIFEST=$DATA_DIR/splits/case_level_split_manifest.csv
    IMAGE_ROOT="${RAW_IMAGE_ROOT:-$PROJECT_ROOT/raw_data/wound_ml_7classes}" ;;
  roi)
    MANIFEST=$DATA_DIR/roi/roi_manifest.csv
    IMAGE_ROOT="$DERIVED_DIR/roi_union_p008" ;;
  box|crop)
    MANIFEST=$DATA_DIR/roi_box/roi_manifest.csv
    IMAGE_ROOT="$DERIVED_DIR/roi_box_p008" ;;
  *)
    echo "INPUT_PATH must be full, roi, box or crop, got: $INPUT_PATH" >&2; exit 1 ;;
esac
# T19: AUTO_ARM=4Z|4P|4F trains on that arm's auto-box crops (union crops for roi, per-box for box and crop).
if [[ -n "${AUTO_ARM:-}" ]]; then
  case "$INPUT_PATH" in
    roi) DESIGN=union ;;
    box|crop) DESIGN=box ;;
    *) echo "AUTO_ARM needs INPUT_PATH roi, box or crop, got: $INPUT_PATH" >&2; exit 1 ;;
  esac
  MANIFEST=$DATA_DIR/roi_auto/$AUTO_ARM/$DESIGN/roi_manifest.csv
  IMAGE_ROOT="$DERIVED_DIR/roi_auto_${AUTO_ARM}_$DESIGN"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENTS_DIR/$RUN_NAME/$INPUT_PATH/$FAMILY}"
export TORCH_HOME="$PROJECT_ROOT/torch_cache"
export PYTHONUNBUFFERED=1  # log lines reach the .out/.err files immediately

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Family: $FAMILY | Input: $INPUT_PATH | Run: $RUN_NAME | Extra: ${EXTRA_ARGS:-none}"
echo "Data: $DATA_DIR | Output: $OUTPUT_DIR"
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
