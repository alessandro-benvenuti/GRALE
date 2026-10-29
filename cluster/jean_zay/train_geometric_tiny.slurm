#!/usr/bin/env bash
# Submit through submit_geometric_tiny.sh, which freezes and checks the targets.
#SBATCH --job-name=grale-octa-l1-200k
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=19:30:00

set -eo pipefail
: "${GRALE_REPO_DIR:?Submit with submit_geometric_tiny.sh}"
: "${GRALE_GEOMETRIC_MANIFEST:?Missing prepared OCTA manifest}"
: "${GRALE_TINY_OUTPUT:?Missing output directory}"
source "$GRALE_REPO_DIR/cluster/jean_zay/env.sh" h100
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python cluster/jean_zay/check_environment.py --gpu \
  --output "$GRALE_TINY_OUTPUT/environment-gpu-$SLURM_JOB_ID.json"
srun "$GRALE_VENV/bin/python" -u scripts/run_geometric.py \
  --mode tiny --config octa_straight_l1_200k --device cuda \
  --steps 200000 --max-time 00:19:00:00 \
  --manifest "$GRALE_GEOMETRIC_MANIFEST" --output "$GRALE_TINY_OUTPUT" \
  --checkpoint-every 10000 --snapshot-every 10000
python - "$GRALE_TINY_OUTPUT/tiny/octa_straight_l1_200k/result.json" <<'PY'
import json
from pathlib import Path
import sys
result = json.loads(Path(sys.argv[1]).read_text())
if result['steps'] != 200000:
    raise SystemExit(f"Training stopped at {result['steps']} / 200000 steps; checkpoint and metrics were retained.")
print('Completed all 200000 training steps. Inspect metrics.json for reconstruction quality.')
PY
