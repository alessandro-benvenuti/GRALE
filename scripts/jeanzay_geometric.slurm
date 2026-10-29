#!/usr/bin/env bash
#SBATCH --job-name=grale-straight
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=logs/geometric-fit-20261029/%x-%j.out
#SBATCH --error=logs/geometric-fit-20261029/%x-%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/logs/geometric-fit-20261029/mpl"
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
mkdir -p "$MPLCONFIGDIR"

srun python -u scripts/run_geometric.py --mode tiny --steps 1000 --device cuda \
    --config "${1:-octa_straight}" --output logs/geometric-fit-20261029 "${@:2}"
