#!/bin/bash
# T11: 2-epoch run of one family on the real images, full and ROI, one H100 each.
# Run on Raptor from the repo root:  bash scripts/hpc/submit_t11_smoke.sh [family]
set -euo pipefail

FAMILY="${1:-efficientnet_b0}"
for input_path in full roi; do
  sbatch -J "t11_${FAMILY}_${input_path}" --time=01:00:00 \
    --export=ALL,FAMILY="$FAMILY",INPUT_PATH="$input_path",RUN_NAME=T11_smoke,EXTRA_ARGS="--max-epochs 2" \
    scripts/slurm/train_single.sbatch
done
