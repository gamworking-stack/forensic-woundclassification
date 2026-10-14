#!/bin/bash
# T03 live run: crop padded union ROIs from the raw images on Raptor.
# CPU only. Submit from the repo root:  sbatch scripts/slurm/t03_roi_crops.sbatch
# T17 per-box crops (one crop per box):
#   sbatch --export=ALL,CROP_MODE=individual,OUTPUT_DIR=/project/a260012_FORWCL/derived_data/roi_box_p008,EXPECTED_MANIFEST=data/roi_box/roi_manifest.csv scripts/slurm/t03_roi_crops.sbatch
#SBATCH --job-name=t03_roi_crops
#SBATCH --account=a260012
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
IMAGE_ROOT="${IMAGE_ROOT:-$PROJECT_ROOT/raw_data/wound_ml_7classes}"
EXPORTS_DIR="${EXPORTS_DIR:-$PROJECT_ROOT/annotations/cvat_yolo_2026_09_30}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/derived_data/roi_union_p008}"
CROP_MODE="${CROP_MODE:-union}"
EXPECTED_MANIFEST="${EXPECTED_MANIFEST:-data/roi/roi_manifest.csv}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "Images: $IMAGE_ROOT"
echo "Exports: $EXPORTS_DIR"
echo "Output: $OUTPUT_DIR (crop mode $CROP_MODE, expected $EXPECTED_MANIFEST)"

for d in "$IMAGE_ROOT" "$EXPORTS_DIR"; do
  [[ -d "$d" ]] || { echo "Missing directory: $d" >&2; exit 1; }
done
echo "Annotation zips: $(find "$EXPORTS_DIR" -maxdepth 1 -name '*.zip' | wc -l) (expected 7)"

module purge
module load apptainer/1.3.5

run() {
  apptainer exec --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" "$SIF" "$@"
}

run python -c "import pandas, PIL, pytest" || {
  echo "pandas/Pillow/pytest missing in $SIF. On the login node run once:" >&2
  echo "  module load apptainer/1.3.5 && apptainer exec $SIF pip install --user pandas pillow pytest" >&2
  exit 1
}

run python -m pytest -q tests/test_roi_preprocessing.py

run python src/data/create_roi_dataset.py \
  --manifest data/splits/case_level_split_manifest.csv \
  --exports "$EXPORTS_DIR" \
  --image-root "$IMAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --padding 0.08 \
  --min-size 8 \
  --crop-mode "$CROP_MODE" \
  --clear-output

run python src/data/compare_roi_manifests.py \
  --expected "$EXPECTED_MANIFEST" \
  --actual "$OUTPUT_DIR/roi_manifest.csv" \
  --crop-root "$OUTPUT_DIR"

echo "Done at $(date -Is)"
