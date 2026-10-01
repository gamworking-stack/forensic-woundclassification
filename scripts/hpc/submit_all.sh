#!/bin/bash
# Submit every S01-S16 scenario that is not finished and not already queued; prints the job IDs.
# Run on Raptor from the repo root:
#   bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_all.sh)
# Finished = test_predictions.csv exists. Unfinished runs resume from last.pt.
set -euo pipefail

RUN_NAME="${1:-main}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
EXP="$PROJECT_ROOT/experiments/$RUN_NAME"
INPUT_PATHS=(full roi)
# Spec §5.1 order; S13-S16 (cnn_svm, alexnet_ensemble) are handled below.
FAMILIES=(efficientnet_b0 resnet152 mobilenet_v3_small googlenet vgg16 vit_b16)

queued=$(squeue -u "$USER" -h -o %j)
ids=()

skip_reason() {  # <family folder> <input path> <job name>
  if [[ -f "$EXP/$2/$1/test_predictions.csv" ]]; then echo finished
  elif grep -qx "$3" <<<"$queued"; then echo queued
  fi
}

submit_single() {  # <scenario id> <family> <input path> <hours>
  local name="${1}_${2}_${3}" reason
  reason=$(skip_reason "$2" "$3" "$name")
  if [[ -n "$reason" ]]; then echo "skip $name: $reason" >&2; return; fi
  ids+=("$(sbatch --parsable -J "$name" --time="$4:00:00" \
    --export=ALL,FAMILY="$2",INPUT_PATH="$3",RUN_NAME="$RUN_NAME" scripts/slurm/train_single.sbatch)")
  echo "submitted $name" >&2
}

for i in "${!FAMILIES[@]}"; do
  for j in 0 1; do
    submit_single "$(printf 'S%02d' $(( 2 * i + j + 1 )))" "${FAMILIES[$i]}" "${INPUT_PATHS[$j]}" 12
  done
done
for j in 0 1; do
  submit_single "$(printf 'S%02d' $(( 13 + j )))" cnn_svm "${INPUT_PATHS[$j]}" 4
done
for j in 0 1; do
  path="${INPUT_PATHS[$j]}" name="alexnet_ensemble_${INPUT_PATHS[$j]}"
  reason=$(skip_reason alexnet_ensemble "$path" "$name")
  if [[ -n "$reason" ]]; then echo "skip $(printf 'S%02d' $(( 15 + j )))_$name: $reason" >&2; continue; fi
  read -r -a new_ids <<<"$(bash scripts/hpc/submit_alexnet_ensemble.sh "$path" "$RUN_NAME")"
  ids+=("${new_ids[@]}")
done

if [[ ${#ids[@]} -eq 0 ]]; then
  echo "Nothing to submit: every scenario is finished or queued." >&2
fi
echo "${ids[*]}"
