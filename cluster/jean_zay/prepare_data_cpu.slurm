#!/usr/bin/env bash
#SBATCH --job-name=grale-prepare
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
set -eo pipefail
repo_dir="${GRALE_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh" "${GRALE_ARCH:-a100}" cpu
set -u
args=(--config "$GRALE_DATA_CONFIG" --dataset-root "$GRALE_DATASET_ROOT"
      --output-dir "$GRALE_PREPARE_OUTPUT")
if [[ "${GRALE_PREPARE_MODE:-smoke}" == smoke ]]; then
  args+=(--smoke-count 8)
fi
# Test graph content is deliberately not an option on the development launcher.
python -u "$repo_dir/scripts/prepare_vascular_pilot.py" "${args[@]}"
