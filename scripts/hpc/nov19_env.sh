# T38: point every script at the 19 Nov 2026 dataset (19 Nov CVAT annotation, new photos, split by patient)
# and its own output folders. Source it (Raptor or laptop) from the repo root before any T38 command:
#   source scripts/hpc/nov19_env.sh
# The dataset's file lists hold forensic case numbers in file names, so they live outside the repo:
# ../datasets/relabel_2026_11_19 on the laptop, <project>/datasets/relabel_2026_11_19 on Raptor.
# The T23 and original folders are not touched.

# The repo root from git, so bash and zsh agree (zsh has no BASH_SOURCE)
T38_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
T38_PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"

if [[ -d "$T38_PROJECT_ROOT" ]]; then   # Raptor
  export WOUND_DATA_DIR="$T38_PROJECT_ROOT/datasets/relabel_2026_11_19"
else                                    # laptop
  export WOUND_DATA_DIR="$(cd "$T38_REPO_ROOT/.." && pwd)/datasets/relabel_2026_11_19"
fi
export EXPERIMENTS_DIR="$T38_PROJECT_ROOT/experiments_relabel_2026_11_19"      # runs, Raptor
export DERIVED_DIR="$T38_PROJECT_ROOT/derived_data/relabel_2026_11_19"         # crops, Raptor
export EXPORTS_DIR="$T38_PROJECT_ROOT/annotations/cvat_2026_11_19"           # 19 Nov CVAT exports, Raptor
export RESULTS_DIR="$T38_REPO_ROOT/../results_figures/relabel_2026_11_19"      # fetched results, laptop

echo "T38 19 Nov dataset: data $WOUND_DATA_DIR | boxes $EXPORTS_DIR | runs $EXPERIMENTS_DIR | crops $DERIVED_DIR" >&2
