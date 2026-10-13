#!/bin/bash
# Copy the aggregate results of a run from Raptor to the laptop, for the thesis figures and tables.
# Usage (laptop, on the CMU network):  bash scripts/hpc/fetch_results.sh [run_name]
# Per-image prediction files, checkpoints, features and images stay on Raptor.
set -euo pipefail

RUN_NAME="${1:-main}"
HPC_USER="${HPC_USER:-preenapun.s}"
HPC_HOST="${HPC_HOST:-raptor.med.cmu.ac.th}"
REMOTE_EXP="${REMOTE_EXP:-/project/a260012_FORWCL/experiments/$RUN_NAME}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_RESULTS="${LOCAL_RESULTS:-$REPO_ROOT/../results_figures/$RUN_NAME}"  # outside the repo, never committed

mkdir -p "$LOCAL_RESULTS"
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='evaluation_index.csv' --include='summary/***' \
  --include='*_metrics.json' --include='*_classification_report.csv' \
  --include='*_confusion_matrix.csv' --include='*_confusion_matrix.png' \
  --include='*_roc_curves.csv' --include='*_roc_curves.png' \
  --include='tuning_summary.csv' --include='ensemble_members.csv' \
  --include='svm_validation_tuning.csv' --include='training_history.csv' --include='run_config.json' \
  --exclude='*' \
  "$HPC_USER@$HPC_HOST:$REMOTE_EXP/" "$LOCAL_RESULTS/"
echo "Results of run '$RUN_NAME' in $LOCAL_RESULTS" >&2
