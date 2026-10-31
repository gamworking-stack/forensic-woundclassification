# T30 "more contrast" experiment: the T23 dataset (manifests, split, boxes) with CLAHE-enhanced images.
# Source it (Raptor or laptop) from the repo root before any T30 command:
#   source scripts/hpc/contrast_env.sh
# It sources relabel_env.sh first, then points images, crops, runs and results at the T30 folders.
# The T23 runs, crops and results are not touched.

source scripts/hpc/relabel_env.sh

T30_PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"
export DERIVED_DIR="$T30_PROJECT_ROOT/derived_data/relabel_2026_10_23_more_contrast"        # CLAHE photos and crops, Raptor
export RAW_IMAGE_ROOT="$DERIVED_DIR/raw_clahe"                                             # full-image input (train_single.sbatch)
export EXPERIMENTS_DIR="$T30_PROJECT_ROOT/experiments_relabel_2026_10_23_more_contrast"    # runs main/ and roi_crop/, Raptor
export RESULTS_DIR="$T23_REPO_ROOT/../results_figures/More contrast experiment"           # results, laptop

echo "T30 more contrast: photos $RAW_IMAGE_ROOT | crops $DERIVED_DIR/roi_box_p008 | runs $EXPERIMENTS_DIR" >&2
