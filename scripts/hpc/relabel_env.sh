# T23: point every script at the 23 Oct 2026 relabelled dataset and its own output folders.
# Source it (Raptor or laptop) from the repo root before any T23 command:
#   source scripts/hpc/relabel_env.sh
# The original data/, experiments/, derived_data/ and ../results_figures/<run> are not touched.

# The repo root from git, so bash and zsh agree (zsh has no BASH_SOURCE)
T23_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
T23_PROJECT_ROOT="${PROJECT_ROOT:-/project/a260012_FORWCL}"

export WOUND_DATA_DIR=data/relabel_2026_10_23                                  # manifests (Python and SLURM)
export EXPERIMENTS_DIR="$T23_PROJECT_ROOT/experiments_relabel_2026_10_23"      # runs, Raptor
export DERIVED_DIR="$T23_PROJECT_ROOT/derived_data/relabel_2026_10_23"         # crops, Raptor
export EXPORTS_DIR="$T23_PROJECT_ROOT/annotations/cvat_2026_10_23"           # 23 Oct CVAT exports, Raptor
export RESULTS_DIR="$T23_REPO_ROOT/../results_figures/relabel_2026_10_23"      # fetched results, laptop

echo "T23 relabelled dataset: data $WOUND_DATA_DIR | boxes $EXPORTS_DIR | runs $EXPERIMENTS_DIR | crops $DERIVED_DIR" >&2
