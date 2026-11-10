#!/bin/bash
# T34: generate cut, scratch and penetration crops from a trained generator, or check their realism
# and memorisation. Submit from the repo root after `source scripts/hpc/relabel_env.sh`:
#   sbatch --export=ALL,MODE=generate,GENERATOR=stylegan2,CHECKPOINT=<network-snapshot.pkl> scripts/slurm/generated_crops.sbatch
#   sbatch --export=ALL,MODE=generate,GENERATOR=sd15_lora,CHECKPOINT=<LoRA folder> scripts/slurm/generated_crops.sbatch
#   sbatch --export=ALL,MODE=quality,SAMPLES="<samples dir> <samples dir>" scripts/slurm/generated_crops.sbatch
#   sbatch --export=ALL,MODE=rating,SAMPLES="<samples dir> <samples dir>" scripts/slurm/generated_crops.sbatch   # blinded set
# Optional: PER_CLASS (default 500), OUT (output folder; default beside the checkpoint, or
# $EXPERIMENTS_DIR/generators/quality). Everything written stays on Raptor (patient-derived).
#SBATCH --job-name=t34_generated
#SBATCH --account=a260012
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${DERIVED_DIR:?source scripts/hpc/relabel_env.sh first}"
: "${MODE:?set MODE to generate or quality}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
TOOLS="$PROJECT_ROOT/tools"
HF_HOME="$PROJECT_ROOT/torch_cache/hf"
BASE="$HF_HOME/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DETECTOR="$PROJECT_ROOT/torch_cache/dnnlib/downloads/18d9c1159d16cd4cc6adf7db0f2dd2a9_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan3_versions_1_files_metrics_inception-2015-12-05.pkl"
TRAIN_DIR="$DERIVED_DIR/generator_crops_256"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"

module purge
module load apptainer/1.3.5

# The diffusion venv sits on the container's Python and PyTorch, so it runs every step
run() {
  apptainer exec --nv --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" \
    --env "TORCH_HOME=$PROJECT_ROOT/torch_cache,TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/torch_cache/extensions,HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1" \
    "$SIF" "$TOOLS/venv_diffusers/bin/python" "$@"
}

run -m pytest -q tests/test_generated_quality.py tests/test_rating_set.py
case "$MODE" in
  generate)
    : "${GENERATOR:?set GENERATOR to stylegan2 or sd15_lora}"
    : "${CHECKPOINT:?set CHECKPOINT}"
    OUT="${OUT:-$(dirname "$CHECKPOINT")/samples_$(basename "${CHECKPOINT%.pkl}")}"
    run src/evaluation/generate_crops.py --generator "$GENERATOR" --checkpoint "$CHECKPOINT" \
      --dataset-info "$TRAIN_DIR/dataset_info.json" --per-class "${PER_CLASS:-500}" --output-dir "$OUT" \
      --stylegan-code "$TOOLS/stylegan3" --base "$BASE" ;;
  quality)
    : "${SAMPLES:?set SAMPLES to one or more samples folders}"
    # shellcheck disable=SC2086
    run src/evaluation/generated_quality.py --samples $SAMPLES --train-dir "$TRAIN_DIR" \
      --image-root "$DERIVED_DIR/roi_box_p008" --detector "$DETECTOR" --stylegan-code "$TOOLS/stylegan3" \
      --output-dir "${OUT:-$EXPERIMENTS_DIR/generators/quality}" ;;
  rating)
    : "${SAMPLES:?set SAMPLES to the samples folders of the quality check}"
    Q="$EXPERIMENTS_DIR/generators/quality"
    # shellcheck disable=SC2086
    run src/evaluation/rating_set.py make --samples $SAMPLES --nearest "$Q/nearest.csv" --train-dir "$TRAIN_DIR" \
      --output-dir "${OUT:-$Q/rating_set}" --key-dir "$Q/rating_key" ;;
  *)
    echo "MODE must be generate, quality or rating, got: $MODE" >&2; exit 1 ;;
esac

echo "Done at $(date -Is)"
