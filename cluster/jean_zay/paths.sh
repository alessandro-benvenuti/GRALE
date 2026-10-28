#!/usr/bin/env bash
# Source from Bash. Repositories/records/logs live in WORK; environments and
# rebuildable exports/caches live in SCRATCH. Never reuse the GNBM environment.
if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "WORK and SCRATCH must be defined by Jean Zay." >&2
  return 2 2>/dev/null || exit 2
fi
export GRALE_REPO_DIR="${GRALE_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export GRALE_ARCH="${GRALE_ARCH:-a100}"
case "$GRALE_ARCH" in
  a100) export GRALE_TORCH_VERSION=2.3.0; grale_env_suffix=a100-torch230 ;;
  h100) export GRALE_TORCH_VERSION=2.3.1; grale_env_suffix=h100-torch231 ;;
  *) echo "GRALE_ARCH must be a100 or h100." >&2; return 2 2>/dev/null || exit 2 ;;
esac
export GRALE_VENV="${GRALE_VENV:-$SCRATCH/venvs/grale-$grale_env_suffix}"
unset grale_env_suffix
case "$GRALE_VENV/" in
  "$SCRATCH/"*) ;;
  *) echo "GRALE_VENV must be inside SCRATCH: $GRALE_VENV" >&2; return 2 2>/dev/null || exit 2 ;;
esac
export GRALE_DATASET_ROOT="${GRALE_DATASET_ROOT:-/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary}"
export GRALE_DATA_CONFIG="${GRALE_DATA_CONFIG:-$GRALE_REPO_DIR/configs/vascular/pilot_data.json}"
export GRALE_PREPARED_DATA="${GRALE_PREPARED_DATA:-$SCRATCH/datasets/grale/synthetic-mri-pilot-v1}"
export GRALE_LOG_DIR="${GRALE_LOG_DIR:-$WORK/logs/grale}"
export GRALE_OUTPUT_DIR="${GRALE_OUTPUT_DIR:-$WORK/experiments/grale-vascular-pilot}"
export XDG_CACHE_HOME="$SCRATCH/.cache/grale"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_DATA_DIR="$SCRATCH/.local/share/grale-wandb"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export TMPDIR="${JOBSCRATCH:-$SCRATCH/tmp/grale-${USER:-user}}"
export WANDB_MODE=offline
export WANDB_PROJECT=grale-vascular
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export IDR_DEBUG="${IDR_DEBUG:-WARN}"
