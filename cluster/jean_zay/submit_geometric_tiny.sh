#!/usr/bin/env bash
# Login-node preflight and submission. Packages are installed separately.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GRALE_REPO_DIR="$repo_dir"
export GRALE_ARCH=h100
source "$repo_dir/cluster/jean_zay/env.sh" h100
set -u

export GRALE_GEOMETRIC_DATASET_ROOT="${GRALE_GEOMETRIC_DATASET_ROOT:-/lustre/fsn1/projects/rech/vnc/upz25mj/datasets}"
manifest_relative=gnbm-native-2d-20261027/experiments/native2d_long_20261028/octasynth/manifest.json
export GRALE_GEOMETRIC_MANIFEST="${1:-${GRALE_GEOMETRIC_MANIFEST:-}}"
if [[ -z "$GRALE_GEOMETRIC_MANIFEST" ]]; then
  # The research audit records the original checkout under upz25mj. Reuse its
  # prepared targets when readable rather than regenerating or copying them.
  for candidate in "$GRALE_GEOMETRIC_DATASET_ROOT/$manifest_relative" \
      "$repo_dir/../GNBM/$manifest_relative" \
      "/lustre/fswork/projects/rech/vnc/upz25mj/projects/GNBM/$manifest_relative"; do
    if [[ -r "$candidate" ]]; then
      GRALE_GEOMETRIC_MANIFEST="$candidate"
      break
    fi
  done
fi
if [[ -z "$GRALE_GEOMETRIC_MANIFEST" || ! -r "$GRALE_GEOMETRIC_MANIFEST" ]]; then
  echo "Missing prepared OCTA manifest: $GRALE_GEOMETRIC_MANIFEST" >&2
  echo "Checked $GRALE_GEOMETRIC_DATASET_ROOT, the sibling GNBM checkout and the original upz25mj preparation." >&2
  echo "The raw dataset directory alone is not a prepared NPZ/ZIP manifest." >&2
  echo "Pass its actual path: bash cluster/jean_zay/submit_geometric_tiny.sh /path/to/octasynth/manifest.json" >&2
  exit 2
fi
GRALE_GEOMETRIC_MANIFEST="$(cd "$(dirname "$GRALE_GEOMETRIC_MANIFEST")" && pwd)/$(basename "$GRALE_GEOMETRIC_MANIFEST")"
export GRALE_GEOMETRIC_MANIFEST
export GRALE_TINY_OUTPUT="${GRALE_TINY_OUTPUT:-$WORK/experiments/grale-octa-tiny-l1-max20-200k/$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$GRALE_TINY_OUTPUT" ]]; then
  echo "Choose a fresh GRALE_TINY_OUTPUT; path already exists: $GRALE_TINY_OUTPUT" >&2
  exit 2
fi
mkdir -p "$GRALE_TINY_OUTPUT" "$GRALE_LOG_DIR"
GRALE_TINY_OUTPUT="$(cd "$GRALE_TINY_OUTPUT" && pwd)"
export GRALE_TINY_OUTPUT
python "$repo_dir/cluster/jean_zay/check_environment.py" \
  --output "$GRALE_TINY_OUTPUT/environment-login.json"
# Explicit cuda selects the intended batch size but --check-inputs never uses a GPU.
python "$repo_dir/scripts/run_geometric.py" --check-inputs --mode tiny \
  --config octa_straight_l1_200k --device cuda --steps 200000 \
  --max-nodes 20 \
  --max-time 00:19:00:00 --manifest "$GRALE_GEOMETRIC_MANIFEST" --output "$GRALE_TINY_OUTPUT"
echo "Training: 200000 steps, learned L1 Sinkhorn, batch size 4, patches with <=20 nodes, one H100."
echo "Source dataset root: $GRALE_GEOMETRIC_DATASET_ROOT"
echo "Prepared manifest: $GRALE_GEOMETRIC_MANIFEST"
echo "Outputs: $GRALE_TINY_OUTPUT"
sbatch --parsable --export=ALL --chdir="$repo_dir" \
  --output="$GRALE_LOG_DIR/%x-%j.out" --error="$GRALE_LOG_DIR/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/train_geometric_tiny.slurm" | tee "$GRALE_TINY_OUTPUT/job-id.txt"
