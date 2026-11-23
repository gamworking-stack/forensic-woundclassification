# T39: the 19 Nov 2026 dataset with cut and penetration wounds merged into "sharp injury" (6 classes),
# with its own split, runs, crops and results. Source it (Raptor or laptop) from the repo root:
#   source scripts/hpc/sharp_env.sh
# Like T38, the dataset's file lists stay outside the repo (case numbers in file names).
# The photos and the 19 Nov CVAT exports are T38's; the T38, T23 and original folders are not touched.

T39_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
T39_PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"

if [[ -d "$T39_PROJECT_ROOT" ]]; then   # Raptor
  export WOUND_DATA_DIR="$T39_PROJECT_ROOT/datasets/relabel_2026_11_19_sharp"
else                                    # laptop
  export WOUND_DATA_DIR="$(cd "$T39_REPO_ROOT/.." && pwd)/datasets/relabel_2026_11_19_sharp"
fi
export WOUND_N_CLASSES=6
export EXPERIMENTS_DIR="$T39_PROJECT_ROOT/experiments_relabel_2026_11_19_sharp"      # runs, Raptor
export DERIVED_DIR="$T39_PROJECT_ROOT/derived_data/relabel_2026_11_19_sharp"         # crops, Raptor
export EXPORTS_DIR="$T39_PROJECT_ROOT/annotations/cvat_2026_11_19"                 # the T38 exports
export RESULTS_DIR="$T39_REPO_ROOT/../results_figures/relabel_2026_11_19_sharp"      # fetched results, laptop

echo "T39 sharp injury (6 classes): data $WOUND_DATA_DIR | runs $EXPERIMENTS_DIR | crops $DERIVED_DIR" >&2
