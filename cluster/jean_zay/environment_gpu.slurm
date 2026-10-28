#!/usr/bin/env bash
#SBATCH --job-name=grale-env-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
set -eo pipefail
repo_dir="${GRALE_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh" "${GRALE_ARCH:-a100}"
set -u
python "$repo_dir/cluster/jean_zay/check_environment.py" --gpu \
  --output "$GRALE_OUTPUT_DIR/environment-smoke-${SLURM_JOB_ID}.json"
