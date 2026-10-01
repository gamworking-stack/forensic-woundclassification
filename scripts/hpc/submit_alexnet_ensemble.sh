#!/bin/bash
# T07: train 3 AlexNet members (seeds 43, 44, 45) on one input path, then soft-vote them.
# Run on Raptor from the repo root:
#   bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_alexnet_ensemble.sh roi)
# Prints the 4 job IDs on stdout. Resubmitting resumes unfinished members.
set -euo pipefail

INPUT_PATH="${1:?usage: submit_alexnet_ensemble.sh <full|roi> [run_name]}"
RUN_NAME="${2:-main}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
ENSEMBLE_DIR="$PROJECT_ROOT/experiments/$RUN_NAME/$INPUT_PATH/alexnet_ensemble"

ids=()
for k in 1 2 3; do
  seed=$(( 42 + k ))
  ids+=("$(sbatch --parsable -J "alexnet_m${k}_${INPUT_PATH}" --time=06:00:00 \
    --export=ALL,FAMILY=alexnet,INPUT_PATH="$INPUT_PATH",RUN_NAME="$RUN_NAME",OUTPUT_DIR="$ENSEMBLE_DIR/member_0$k",EXTRA_ARGS="--seed $seed" \
    scripts/slurm/train_single.sbatch)")
done

# kill-on-invalid-dep: if a member fails, cancel the combine job instead of leaving it pending.
combine=$(sbatch --parsable -J "alexnet_ensemble_${INPUT_PATH}" \
  --dependency="afterok:$(IFS=:; echo "${ids[*]}")" --kill-on-invalid-dep=yes \
  --export=ALL,ENSEMBLE_DIR="$ENSEMBLE_DIR" \
  scripts/slurm/ensemble_combine.sbatch)

echo "Members ${ids[*]} -> combine $combine ($ENSEMBLE_DIR)" >&2
echo "${ids[*]} $combine"
