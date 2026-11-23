#!/bin/bash
# Upload the repo and the 7 CVAT exports (annotation text only, no images) to Raptor,
# plus the 7 exports of 23 Oct 2026 for T23 when their folder is present.
# Usage:  bash scripts/hpc/sync_to_raptor.sh
set -euo pipefail

HPC_USER="${HPC_USER:-preenapun.s}"
HPC_HOST="${HPC_HOST:-raptor.med.cmu.ac.th}"
REMOTE_REPO="${REMOTE_REPO:-forensic-woundclassification}"
REMOTE_EXPORTS="${REMOTE_EXPORTS:-/project/a260012_FORWCL/annotations/cvat_yolo_2026_09_30}"
LOCAL_EXPORTS="${LOCAL_EXPORTS:-$HOME/Downloads}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$HPC_USER@$HPC_HOST"

# Runs record the commit they ran from (run_config.json git_commit/git_dirty), so only a
# committed tree is uploaded. ALLOW_DIRTY=1 overrides this for a throwaway test.
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" && "${ALLOW_DIRTY:-0}" != 1 ]]; then
  echo "Uncommitted changes in $REPO_ROOT; commit them first (or ALLOW_DIRTY=1 for a test):" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 1
fi

# The 30 Sep exports used for the committed ROI manifest; browser copies like "(1).zip" are skipped.
exports=()
for task in 12 14 15 16 17 18 19; do
  match=("$LOCAL_EXPORTS"/task_${task}_annotations_2026_09_30_*_yolo\ 1.1.zip)
  [[ -f "${match[0]}" ]] || { echo "Export for task $task not found in $LOCAL_EXPORTS" >&2; exit 1; }
  exports+=("${match[0]}")
done

rsync -av --exclude '__pycache__/' --exclude '.venv/' --exclude '.DS_Store' \
  "$REPO_ROOT/" "$DEST:$REMOTE_REPO/"

ssh "$DEST" "mkdir -p '$REMOTE_EXPORTS'"
rsync -av "${exports[@]}" "$DEST:$REMOTE_EXPORTS/"

# T23: the 23 Oct 2026 exports (one per class; laceration as CVAT XML) behind data/relabel_2026_10_23
T23_LOCAL_EXPORTS="${T23_LOCAL_EXPORTS:-$REPO_ROOT/../23 oct updated annotation CVAT}"
T23_REMOTE_EXPORTS="${T23_REMOTE_EXPORTS:-/project/a260012_FORWCL/annotations/cvat_2026_10_23}"
if [[ -d "$T23_LOCAL_EXPORTS" ]]; then
  ssh "$DEST" "mkdir -p '$T23_REMOTE_EXPORTS'"
  rsync -av "$T23_LOCAL_EXPORTS"/*.zip "$DEST:$T23_REMOTE_EXPORTS/"
fi

# T38: the 19 Nov 2026 exports, the dataset lists (kept outside the repo: file names hold case numbers)
# and the photos that are not on Raptor yet, when the dataset folder has been prepared
T38_LOCAL_DATA="${T38_LOCAL_DATA:-$REPO_ROOT/../datasets/relabel_2026_11_19}"
T38_LOCAL_EXPORTS="${T38_LOCAL_EXPORTS:-$REPO_ROOT/../19 Nov annotation}"
T38_REMOTE_DATA="${T38_REMOTE_DATA:-/project/a260012_FORWCL/datasets/relabel_2026_11_19}"
T38_REMOTE_EXPORTS="${T38_REMOTE_EXPORTS:-/project/a260012_FORWCL/annotations/cvat_2026_11_19}"
RAW_TREE="${RAW_TREE:-/project/a260012_FORWCL/raw_data/wound_ml_7classes}"
if [[ -f "$T38_LOCAL_DATA/validated_metadata.csv" ]]; then
  ssh "$DEST" "mkdir -p '$T38_REMOTE_DATA' '$T38_REMOTE_EXPORTS'"
  rsync -av "$T38_LOCAL_EXPORTS"/*.zip "$DEST:$T38_REMOTE_EXPORTS/"
  rsync -av --exclude '.DS_Store' "$T38_LOCAL_DATA/" "$DEST:$T38_REMOTE_DATA/"
  # T39: the same photos and exports with cut and penetration merged (6 classes), its own lists
  T39_LOCAL_DATA="${T39_LOCAL_DATA:-$REPO_ROOT/../datasets/relabel_2026_11_19_sharp}"
  if [[ -f "$T39_LOCAL_DATA/validated_metadata.csv" ]]; then
    ssh "$DEST" "mkdir -p '$T38_REMOTE_DATA/../relabel_2026_11_19_sharp'"
    rsync -av --exclude '.DS_Store' "$T39_LOCAL_DATA/" "$DEST:$T38_REMOTE_DATA/../relabel_2026_11_19_sharp/"
  fi
  # New photos go into the raw tree under their raw label folder; an existing file is never replaced.
  # They are staged as links in one folder per label, so one rsync sends them all
  STAGE="$(mktemp -d)"
  trap 'rm -rf "$STAGE"' EXIT
  python3 - "$T38_LOCAL_DATA/raptor_upload.csv" "$HOME/Desktop" "$STAGE" <<'PY'
import csv, os, sys
listing, root, stage = sys.argv[1:]
for row in csv.DictReader(open(listing, encoding="utf-8")):
    src = os.path.join(root, row["local_file"])
    if not os.path.isfile(src):
        sys.exit(f"Photo not found: {src}")
    folder = os.path.join(stage, row["source_label"])
    os.makedirs(folder, exist_ok=True)
    os.symlink(src, os.path.join(folder, os.path.basename(src)))
PY
  for folder in "$STAGE"/*/; do
    label="$(basename "$folder")"
    ssh "$DEST" "test -d '$RAW_TREE/$label'" || { echo "No raw folder $RAW_TREE/$label on Raptor" >&2; exit 1; }
    rsync -avL --ignore-existing "$folder" "$DEST:$RAW_TREE/$label/"
  done
fi
