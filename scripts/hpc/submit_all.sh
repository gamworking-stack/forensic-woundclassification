#!/bin/bash
# Submit every S01-S16 scenario that is not finished and not already queued; prints the job IDs.
# Run on Raptor from the repo root:
#   bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_all.sh)
# Finished = test_predictions.csv exists. Unfinished runs resume from last.pt.
# S01-S12 are tuned (T08b): one run per learning rate in lr_<value>/, then a select job.
# T17 per-box run (S17-S24):  PATHS=box bash scripts/hpc/submit_all.sh roi_box
set -euo pipefail

RUN_NAME="${1:-main}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
EXP="$PROJECT_ROOT/experiments/$RUN_NAME"
read -r -a INPUT_PATHS <<<"${PATHS:-full roi}"
# Spec §5.1 order; S13-S16 (cnn_svm, alexnet_ensemble) are handled below.
FAMILIES=(efficientnet_b0 resnet152 mobilenet_v3_small googlenet vgg16 vit_b16)
LRS=(1e-5 3e-5 1e-4)  # T08b grid, fixed before any S01-S12 run; same as select_lr.py LR_GRID

queued=$(squeue -u "$USER" -h -o %j)
ids=()

scenario_id() {  # <family index 0-7> <input path>: S01-S16 as in spec §5.1, S17-S24 for box (T17)
  case "$2" in
    full) printf 'S%02d' $(( 2 * $1 + 1 )) ;;
    roi)  printf 'S%02d' $(( 2 * $1 + 2 )) ;;
    box)  printf 'S%02d' $(( 17 + $1 )) ;;
    *)    echo "unknown input path: $2" >&2; exit 1 ;;
  esac
}

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

submit_tuned() {  # <scenario id> <family> <input path> <hours per grid run>
  local name="${1}_${2}_${3}" dir="$EXP/$3/$2" reason lr job id deps=()
  reason=$(skip_reason "$2" "$3" "${name}_select")
  if [[ -n "$reason" ]]; then echo "skip $name: $reason" >&2; return; fi
  for lr in "${LRS[@]}"; do
    job="${name}_lr${lr}"
    if [[ -f "$dir/lr_$lr/validation_predictions.csv" ]]; then continue; fi
    if grep -qx "$job" <<<"$queued"; then
      deps+=("$(squeue -u "$USER" -h -n "$job" -o %i | head -n 1)")
      continue
    fi
    id=$(sbatch --parsable -J "$job" --time="$4:00:00" \
      --export=ALL,FAMILY="$2",INPUT_PATH="$3",RUN_NAME="$RUN_NAME",OUTPUT_DIR="$dir/lr_$lr",EXTRA_ARGS="--lr $lr --skip-test" \
      scripts/slurm/train_single.sbatch)
    ids+=("$id"); deps+=("$id")
  done
  # kill-on-invalid-dep: if a grid run fails, cancel the select job instead of leaving it pending.
  local dep_args=()
  if [[ ${#deps[@]} -gt 0 ]]; then
    dep_args=(--dependency="afterok:$(IFS=:; echo "${deps[*]}")" --kill-on-invalid-dep=yes)
  fi
  ids+=("$(sbatch --parsable -J "${name}_select" ${dep_args[@]+"${dep_args[@]}"} \
    --export=ALL,SCENARIO_DIR="$dir" scripts/slurm/tune_select.sbatch)")
  echo "submitted $name: ${#deps[@]} grid runs + select" >&2
}

for i in "${!FAMILIES[@]}"; do
  for path in "${INPUT_PATHS[@]}"; do
    submit_tuned "$(scenario_id "$i" "$path")" "${FAMILIES[$i]}" "$path" 12
  done
done
for path in "${INPUT_PATHS[@]}"; do
  submit_single "$(scenario_id 6 "$path")" cnn_svm "$path" 4
done
for path in "${INPUT_PATHS[@]}"; do
  name="alexnet_ensemble_${path}"
  reason=$(skip_reason alexnet_ensemble "$path" "$name")
  if [[ -n "$reason" ]]; then echo "skip $(scenario_id 7 "$path")_$name: $reason" >&2; continue; fi
  read -r -a new_ids <<<"$(bash scripts/hpc/submit_alexnet_ensemble.sh "$path" "$RUN_NAME")"
  ids+=("${new_ids[@]}")
done

if [[ ${#ids[@]} -eq 0 ]]; then
  echo "Nothing to submit: every scenario is finished or queued." >&2
fi
echo "${ids[*]}"
