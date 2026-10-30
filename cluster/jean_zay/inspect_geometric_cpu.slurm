#!/usr/bin/env bash
#SBATCH --job-name=grale-inspect
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00

set -eo pipefail
: "${GRALE_REPO_DIR:?Submit with submit_geometric_inspection.sh}"
: "${GRALE_INSPECT_RUN:?Missing run directory}"
: "${GRALE_INSPECT_OUTPUT:?Missing inspection output}"
# Select the same installed Python/Torch module and SCRATCH venv as training.
# Loading arch/h100 selects libraries; Slurm allocates only CPU resources here.
source "$GRALE_REPO_DIR/cluster/jean_zay/env.sh" h100
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python cluster/jean_zay/check_environment.py \
  --output "$GRALE_LOG_DIR/grale-inspect-environment-$SLURM_JOB_ID.json"
srun "$GRALE_VENV/bin/python" -u scripts/inspect_geometric_run.py \
  --run-dir "$GRALE_INSPECT_RUN" --output "$GRALE_INSPECT_OUTPUT"
