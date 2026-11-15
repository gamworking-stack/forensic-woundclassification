#!/bin/bash
# One T37 step (YOLO detector arm 4Y): run PY_ARGS in the container, GPU by default.
#   sbatch -J 4Y_train_m --export=ALL,PY_ARGS="src/detection/yolo.py train ..." scripts/slurm/t37.sbatch
# Several commands: separate them with ' | ' in PY_ARGS; they run in order and the job stops at the first failure.
# CPU steps: add  -p compute --gres=none  to sbatch.
# VENV: venv_exp4_yolo (default; ultralytics 8.4.178) or "none" for the container's own python.
# Unlike exp4.sbatch the job is online: ultralytics' AMP check and fonts may download (compute nodes have internet);
# the COCO weights are local files in /project/a260012_FORWCL/yolo_weights (SHA-256 in tickets/T37_yolo_detector.md).
# Globs in PY_ARGS expand when the job runs (e.g. the one test detections file of the chosen configuration).
#SBATCH --job-name=t37
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
: "${WOUND_DATA_DIR:?source scripts/hpc/relabel_env.sh first}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
SIF="${SIF:-/common/sif/pytorch-24.11-py3.sif}"
VENV="${VENV:-venv_exp4_yolo}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CA=/etc/ssl/certs/ca-certificates.crt
ENVS="WOUND_DATA_DIR=$WOUND_DATA_DIR,YOLO_CONFIG_DIR=$PROJECT_ROOT/yolo_config,TORCH_HOME=$PROJECT_ROOT/torch_cache,MPLCONFIGDIR=/tmp,SSL_CERT_FILE=$CA,REQUESTS_CA_BUNDLE=$CA,PYTHONUNBUFFERED=1"
PY="$PROJECT_ROOT/$VENV/bin/python"
[[ "$VENV" == none ]] && PY=python

cd "$REPO_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is), commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
module purge
module load apptainer/1.3.5
NV=()
[[ -n "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" ]] && NV=(--nv)

IFS='|' read -ra COMMANDS <<< "$PY_ARGS"
for command in "${COMMANDS[@]}"; do
  echo ">>> $PY $command"
  # The command is split into words (and its globs expanded) on purpose.
  # shellcheck disable=SC2086
  apptainer exec "${NV[@]}" --bind "$PROJECT_ROOT" --pwd "$REPO_DIR" --env "$ENVS" "$SIF" $PY $command
done
echo "Done at $(date -Is)"
