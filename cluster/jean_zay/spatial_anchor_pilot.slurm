#!/usr/bin/env bash
#SBATCH --job-name=grale-anchor-pilot
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00

set -eo pipefail
: "${GRALE_REPO_DIR:?Set an isolated source directory}"
: "${GRALE_ANCHOR_OUTPUT:?Set a fresh output path}"
: "${GRALE_ANCHOR_REFERENCE:?Set the completed 200k reference run}"
source "$GRALE_REPO_DIR/cluster/jean_zay/env.sh" h100
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python cluster/jean_zay/check_environment.py --gpu \
  --output "$GRALE_LOG_DIR/grale-anchor-environment-$SLURM_JOB_ID.json"
srun "$GRALE_VENV/bin/python" -u scripts/run_spatial_anchor_pilot.py \
  --reference-run "$GRALE_ANCHOR_REFERENCE" --output "$GRALE_ANCHOR_OUTPUT" \
  --device cuda --steps 5000 --hold-steps 500 --ramp-steps 500
