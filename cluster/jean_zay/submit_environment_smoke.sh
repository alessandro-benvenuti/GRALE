#!/usr/bin/env bash
set -euo pipefail
export GRALE_ARCH="${1:-a100}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
[[ -x "$GRALE_VENV/bin/python" ]] || { echo "Run setup_environment.sh $GRALE_ARCH first." >&2; exit 2; }
mkdir -p "$GRALE_LOG_DIR"
case "$GRALE_ARCH" in
  a100) grale_partition=gpu_p5 ;;
  h100) grale_partition=gpu_p6 ;;
esac
sbatch --export=ALL --chdir="$GRALE_REPO_DIR" --account="vnc@$GRALE_ARCH" \
  --partition="$grale_partition" --constraint="$GRALE_ARCH" --qos="qos_gpu_${GRALE_ARCH}-dev" \
  --output="$GRALE_LOG_DIR/%x-%j.out" --error="$GRALE_LOG_DIR/%x-%j.err" \
  "$GRALE_REPO_DIR/cluster/jean_zay/environment_gpu.slurm"
