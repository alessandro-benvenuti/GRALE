#!/usr/bin/env bash
#SBATCH --job-name=grale-synthetic-matching
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=logs/synthetic-matching-%j.out
#SBATCH --error=logs/synthetic-matching-%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/grale-synthetic-mpl
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
srun python -u scripts/diagnose_synthetic_matching.py --device cuda --steps 2000 "$@"
