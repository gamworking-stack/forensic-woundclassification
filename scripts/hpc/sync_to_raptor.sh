#!/bin/bash
# Upload the repo and the 7 CVAT exports (annotation text only, no images) to Raptor.
# Usage:  bash scripts/hpc/sync_to_raptor.sh
set -euo pipefail

HPC_USER="${HPC_USER:-preenapun.s}"
HPC_HOST="${HPC_HOST:-raptor.med.cmu.ac.th}"
REMOTE_REPO="${REMOTE_REPO:-forensic-woundclassification}"
REMOTE_EXPORTS="${REMOTE_EXPORTS:-/project/a260012_FORWCL/annotations/cvat_yolo_2026_09_30}"
LOCAL_EXPORTS="${LOCAL_EXPORTS:-$HOME/Downloads}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$HPC_USER@$HPC_HOST"

# The 30 Sep exports used for the committed ROI manifest; browser copies like "(1).zip" are skipped.
exports=()
for task in 12 14 15 16 17 18 19; do
  match=("$LOCAL_EXPORTS"/task_${task}_annotations_2026_09_30_*_yolo\ 1.1.zip)
  [[ -f "${match[0]}" ]] || { echo "Export for task $task not found in $LOCAL_EXPORTS" >&2; exit 1; }
  exports+=("${match[0]}")
done

rsync -av --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' --exclude '.DS_Store' \
  "$REPO_ROOT/" "$DEST:$REMOTE_REPO/"

ssh "$DEST" "mkdir -p '$REMOTE_EXPORTS'"
rsync -av "${exports[@]}" "$DEST:$REMOTE_EXPORTS/"
