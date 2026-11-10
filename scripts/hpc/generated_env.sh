# T34 stage 2: the T23 per-crop runs retrained with generated crops added to the train split.
# Source it on Raptor from the repo root with the generator's arm before any T34 retraining command:
#   GENERATED=gan source scripts/hpc/generated_env.sh     # StyleGAN2-ADA, snapshot 2,600 kimg
#   GENERATED=sd source scripts/hpc/generated_env.sh      # Stable Diffusion 1.5 LoRA, final weights
# It sources relabel_env.sh first (real crops, split, boxes unchanged), then sets EXTRA_TRAIN (read by
# train_single.sbatch) and the arm's own run folder. The T23 runs are not touched.

source scripts/hpc/relabel_env.sh

T34_PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
case "${GENERATED:-}" in
  gan|sd) ;;
  *) echo "set GENERATED=gan or GENERATED=sd before sourcing generated_env.sh" >&2; return 1 ;;
esac
export EXTRA_TRAIN="$EXPERIMENTS_DIR/generators/extra_train_$GENERATED.csv"           # from select_generated.py
export EXPERIMENTS_DIR="$T34_PROJECT_ROOT/experiments_relabel_2026_10_23_generated_$GENERATED"   # run roi_crop/, Raptor
export RESULTS_DIR="$T23_REPO_ROOT/../results_figures/Generated wounds/$GENERATED"     # results, laptop

echo "T34 generated ($GENERATED): extra train $EXTRA_TRAIN | runs $EXPERIMENTS_DIR" >&2
