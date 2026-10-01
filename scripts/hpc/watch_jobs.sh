#!/bin/bash
# Follow SLURM job logs (.out and .err) live, and stop when the jobs have finished.
# Usage (Raptor, repo root):  bash scripts/hpc/watch_jobs.sh [jobid ...]
#   With no job IDs, watches all of your queued and running jobs.
#   Ctrl+C stops watching only; the jobs keep running.
set -euo pipefail

ids=("$@")
if [[ ${#ids[@]} -eq 0 ]]; then
  mapfile -t ids < <(squeue -u "$USER" -h -o %i)
fi
if [[ ${#ids[@]} -eq 0 ]]; then
  echo "No queued or running jobs."
  exit 0
fi
id_list=$(IFS=,; echo "${ids[*]}")

files=()
for id in "${ids[@]}"; do
  out=$(scontrol show job "$id" 2>/dev/null | sed -n 's/^ *StdOut=//p')
  [[ -n "$out" ]] || out=$(ls logs/*_"$id".out 2>/dev/null | head -n 1)
  [[ -n "$out" ]] || { echo "No log path found for job $id" >&2; continue; }
  files+=("$out" "${out%.out}.err")
done

echo "Watching jobs $id_list (Ctrl+C stops watching; the jobs keep running)"
tail -n +1 -F "${files[@]}" 2>/dev/null &
tail_pid=$!
trap 'kill "$tail_pid" 2>/dev/null' EXIT

while squeue -h -j "$id_list" 2>/dev/null | grep -q .; do
  sleep 10
done
sleep 2

echo
echo "All jobs finished:"
sacct -j "$id_list" -X --format=JobID,JobName%30,State,ExitCode,Elapsed
