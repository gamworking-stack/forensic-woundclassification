#!/bin/bash
# T34: LoRA fine-tuning of Stable Diffusion 1.5 on the training wound crops (all 7 classes, 512 px,
# one caption per class), one H100. The diffusion counterpart of stylegan_train.sbatch.
# Submit from the repo root after `source scripts/hpc/relabel_env.sh`:
#   sbatch scripts/slurm/diffusion_lora_train.sbatch
#   sbatch --export=ALL,STEPS=20,CKPT=10,RUN=smoke scripts/slurm/diffusion_lora_train.sbatch   # smoke test
# Optional: FOLD (a CV fold, T12b), STEPS (default 15000), CKPT (checkpoint every, default 2500), RANK (default 16).
# Code: huggingface/diffusers v0.31.0 examples/text_to_image/train_text_to_image_lora.py and a venv with
# diffusers 0.31.0 on the container's PyTorch, both in $PROJECT_ROOT/tools; weights
# stable-diffusion-v1-5/stable-diffusion-v1-5 (revision 451f4fe, CreativeML OpenRAIL-M) in the HF cache.
# Writes $DERIVED_DIR/generator_crops_512[_fold<k>]/ and $EXPERIMENTS_DIR/generators/sd15_lora/<run>/ (stay on Raptor).
#SBATCH --job-name=t34_sd_lora
#SBATCH --account=a260012
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${DERIVED_DIR:?source scripts/hpc/relabel_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
TOOLS="$PROJECT_ROOT/tools"
HF_HOME="$PROJECT_ROOT/torch_cache/hf"
BASE="$HF_HOME/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SUFFIX="${FOLD:+_fold$FOLD}"
DATA="$DERIVED_DIR/generator_crops_512$SUFFIX"
RUN="${RUN:-main$SUFFIX}"
OUT="$EXPERIMENTS_DIR/generators/sd15_lora/$RUN"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "Repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git')); diffusers examples $(git -C "$TOOLS/diffusers_examples" rev-parse --short HEAD)"

module purge
module load apptainer/1.3.5

# The venv sits on the container's own Python and PyTorch; the hub is offline (weights are cached)
run() {
  apptainer exec --nv --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" \
    --env "TORCH_HOME=$PROJECT_ROOT/torch_cache,HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1" \
    "$SIF" "$@"
}

run python -m pytest -q tests/test_generator_crops.py
if [[ ! -f "$DATA/metadata.jsonl" ]]; then
  run python src/data/export_generator_crops.py --image-root "$DERIVED_DIR/roi_box_p008" --output-dir "$DATA" --size 512 \
    ${FOLD:+--cv-folds "$WOUND_DATA_DIR/splits/cv_folds.csv" --fold "$FOLD"}
fi
mkdir -p "$OUT"
run "$TOOLS/venv_diffusers/bin/python" "$TOOLS/diffusers_examples/examples/text_to_image/train_text_to_image_lora.py" \
  --pretrained_model_name_or_path "$BASE" --train_data_dir "$DATA" --caption_column text \
  --resolution 512 --random_flip --train_batch_size 8 --dataloader_num_workers 4 \
  --max_train_steps "${STEPS:-15000}" --checkpointing_steps "${CKPT:-2500}" \
  --learning_rate 1e-4 --lr_scheduler cosine --lr_warmup_steps 500 --rank "${RANK:-16}" --snr_gamma 5 \
  --mixed_precision bf16 --seed 42 --output_dir "$OUT"

echo "Done at $(date -Is)"
