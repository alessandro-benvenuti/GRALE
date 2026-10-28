#!/usr/bin/env bash
set -euo pipefail
mode="${1:-smoke}"
case "$mode" in smoke|full) ;; *) echo "Usage: bash $0 smoke|full" >&2; exit 2 ;; esac
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
export GRALE_PREPARE_MODE="$mode"
if [[ "$mode" == smoke ]]; then
  export GRALE_PREPARE_OUTPUT="${GRALE_PREPARE_OUTPUT:-${GRALE_PREPARED_DATA}-smoke}"
else
  export GRALE_PREPARE_OUTPUT="${GRALE_PREPARE_OUTPUT:-$GRALE_PREPARED_DATA}"
fi
[[ -x "$GRALE_VENV/bin/python" ]] || { echo "Run setup_environment.sh $GRALE_ARCH first." >&2; exit 2; }
args=(--check-inputs --config "$GRALE_DATA_CONFIG" --dataset-root "$GRALE_DATASET_ROOT")
if [[ -n "${GRALE_PATIENT_SPLIT_FILE:-}" ]]; then
  args+=(--patient-split-file "$GRALE_PATIENT_SPLIT_FILE")
fi
"$GRALE_VENV/bin/python" "$GRALE_REPO_DIR/scripts/prepare_vascular_pilot.py" "${args[@]}"
if [[ -e "$GRALE_PREPARE_OUTPUT" ]]; then
  echo "Choose a fresh GRALE_PREPARE_OUTPUT; path already exists: $GRALE_PREPARE_OUTPUT" >&2
  exit 2
fi
mkdir -p "$GRALE_LOG_DIR"
echo "Dataset: $GRALE_DATASET_ROOT"
echo "Configuration: $GRALE_DATA_CONFIG"
echo "Mode: $mode; output: $GRALE_PREPARE_OUTPUT"
echo "Train/validation graph export; test manifest only. No GPU requested."
sbatch --export=ALL --chdir="$GRALE_REPO_DIR" \
  --output="$GRALE_LOG_DIR/%x-%j.out" --error="$GRALE_LOG_DIR/%x-%j.err" \
  "$GRALE_REPO_DIR/cluster/jean_zay/prepare_data_cpu.slurm"
