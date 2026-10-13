#!/bin/bash
# T12b: submit every 5-fold CV job that is not finished and not already queued; prints the job IDs.
# Run on Raptor from the repo root, after the main run has finished:
#   bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_cv.sh)
# Finished = validation_predictions.csv (the held-out fold) exists. The test split is never loaded.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
MAIN="$PROJECT_ROOT/experiments/${MAIN_RUN:-main}"
CV="$PROJECT_ROOT/experiments/${CV_RUN:-cv}"
FOLDS_CSV=data/splits/cv_folds.csv
HOURS="${HOURS:-4}"

mkdir -p "$CV"
python3 src/training/cv_plan.py --main-root "$MAIN" --cv-root "$CV" --output "$CV/cv_plan.tsv"

queued=$(squeue -u "$USER" -h -o %j)
ids=()
declare -A member_ids  # fold folder -> colon-separated member job IDs submitted or queued

is_queued() { grep -qx "$1" <<<"$queued"; }
queued_id() { squeue -u "$USER" -h -n "$1" -o %i | head -n 1; }

# Plan fields are tab-separated; "-" marks an empty field.
while IFS=$'\t' read -r name kind family input_path fold lr epochs seed svm_from out; do
  out="${out%$'\r'}"  # tolerate CRLF plan files
  if [[ "$name" == job_name ]]; then continue; fi
  if [[ -f "$out/validation_predictions.csv" ]]; then continue; fi
  fold_dir="${out%/member_*}"
  if is_queued "$name"; then
    if [[ "$kind" == member ]]; then member_ids[$fold_dir]+="$(queued_id "$name"):"; fi
    continue
  fi
  cv_args="--cv-folds $FOLDS_CSV --fold $fold"
  case "$kind" in
    train|member)
      extra="$cv_args --lr $lr --max-epochs $epochs --patience $epochs --checkpoint last --seed $seed" ;;
    svm)
      extra="$cv_args --svm-params-from $svm_from" ;;
    combine)
      deps="${member_ids[$fold_dir]:-}"; deps="${deps%:}"
      dep_args=()
      if [[ -n "$deps" ]]; then dep_args=(--dependency="afterok:$deps" --kill-on-invalid-dep=yes); fi
      ids+=("$(sbatch --parsable -J "$name" ${dep_args[@]+"${dep_args[@]}"} \
        --export=ALL,ENSEMBLE_DIR="$out",ENSEMBLE_SPLITS=validation scripts/slurm/ensemble_combine.sbatch)")
      echo "submitted $name" >&2
      continue ;;
  esac
  id=$(sbatch --parsable -J "$name" --time="$HOURS:00:00" \
    --export=ALL,FAMILY="$family",INPUT_PATH="$input_path",OUTPUT_DIR="$out",EXTRA_ARGS="$extra" \
    scripts/slurm/train_single.sbatch)
  ids+=("$id")
  if [[ "$kind" == member ]]; then member_ids[$fold_dir]+="$id:"; fi
  echo "submitted $name" >&2
done < "$CV/cv_plan.tsv"

if [[ ${#ids[@]} -eq 0 ]]; then
  echo "Nothing to submit: every CV job is finished or queued." >&2
fi
echo "${ids[*]:-}"
