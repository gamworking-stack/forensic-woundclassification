#!/bin/bash
# One T19 (Experiment 4) step: run PY_ARGS with a venv's python in the container (GPU by default).
#   sbatch -J 4Z_gdino_dev --export=ALL,PY_ARGS="src/detection/zero_shot.py --model gdino ..." scripts/slurm/exp4.sbatch
# CPU steps: add  -p compute --gres=none  (or --gres= ) to sbatch.
# VENV: venv_exp4 (default) or venv_exp4_yolo (4P detect).
# READ_ONLY=1 binds /project read-only and only WRITABLE (a folder) read-write: the 4P prepare job.
#SBATCH --job-name=exp4
#SBATCH --account=a260012
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${PY_ARGS:?set PY_ARGS to a script and its arguments}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
VENV="${VENV:-venv_exp4}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CA=/etc/ssl/certs/ca-certificates.crt  # the container's bundle; the host's Red Hat path does not exist there
ENVS="HF_HOME=$PROJECT_ROOT/hf_cache,HF_HUB_OFFLINE=1,YOLO_OFFLINE=1,TORCH_HOME=$PROJECT_ROOT/torch_cache,SSL_CERT_FILE=$CA,REQUESTS_CA_BUNDLE=$CA,PYTHONUNBUFFERED=1"

BINDS=(--bind "$PROJECT_ROOT")
if [[ "${READ_ONLY:-0}" == 1 ]]; then
  : "${WRITABLE:?READ_ONLY=1 needs WRITABLE, the one folder the job may write}"
  mkdir -p "$WRITABLE"
  BINDS=(--bind "$PROJECT_ROOT:$PROJECT_ROOT:ro" --bind "$WRITABLE:$WRITABLE")
fi

cd "$REPO_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is): $VENV python $PY_ARGS"
module purge
module load apptainer/1.3.5
# PY_ARGS is split into words on purpose.
# shellcheck disable=SC2086
apptainer exec --nv "${BINDS[@]}" --pwd "$REPO_DIR" --env "$ENVS" "$SIF" \
  "$PROJECT_ROOT/$VENV/bin/python" $PY_ARGS
echo "Done at $(date -Is)"
