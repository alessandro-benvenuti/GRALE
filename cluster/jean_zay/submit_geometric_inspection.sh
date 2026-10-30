#!/usr/bin/env bash
# Inspect an existing completed run; no training or GPU allocation.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: bash $0 /path/to/run/tiny/octa_straight_l1_200k" >&2
  exit 2
fi
export GRALE_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GRALE_ARCH=h100
source "$GRALE_REPO_DIR/cluster/jean_zay/paths.sh"
export GRALE_INSPECT_RUN="$(cd "$1" && pwd)"
for required in result.json last.ckpt; do
  [[ -r "$GRALE_INSPECT_RUN/$required" ]] || { echo "Missing $GRALE_INSPECT_RUN/$required" >&2; exit 2; }
done
[[ -x "$GRALE_VENV/bin/python" ]] || { echo "Missing H100 environment: $GRALE_VENV" >&2; exit 2; }
export GRALE_INSPECT_OUTPUT="${GRALE_INSPECT_OUTPUT:-$GRALE_INSPECT_RUN/inspection-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$GRALE_INSPECT_OUTPUT" || -e "$GRALE_INSPECT_OUTPUT.zip" ]]; then
  echo "Choose a new GRALE_INSPECT_OUTPUT: $GRALE_INSPECT_OUTPUT" >&2
  exit 2
fi
mkdir -p "$GRALE_LOG_DIR"
echo "CPU inspection of: $GRALE_INSPECT_RUN"
echo "Results: $GRALE_INSPECT_OUTPUT"
echo "Log: $GRALE_LOG_DIR/grale-inspect-JOBID.out"
sbatch --parsable --export=ALL --chdir="$GRALE_REPO_DIR" \
  --output="$GRALE_LOG_DIR/%x-%j.out" --error="$GRALE_LOG_DIR/%x-%j.err" \
  "$GRALE_REPO_DIR/cluster/jean_zay/inspect_geometric_cpu.slurm"
