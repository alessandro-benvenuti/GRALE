#!/usr/bin/env bash
#SBATCH --job-name=grale-matchers
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00

set -eo pipefail
: "${GRALE_REPO_DIR:?Set isolated source directory}"
: "${GRALE_MATCHER_REFERENCE:?Set the frozen reference run}"
: "${GRALE_MATCHER_INSPECTION:?Set the recovered GT inspection}"
: "${GRALE_MATCHER_OUTPUT:?Set fresh output directory}"
: "${GRALE_SOLVER_DEPS:?Set isolated POT package directory in SCRATCH}"
case "${GRALE_MATCHER_ARCH:-a100}" in
  a100) export GRALE_VENV="$SCRATCH/venvs/grale-a100-torch230" ;;
  h100) export GRALE_VENV="$SCRATCH/venvs/grale-h100-torch231" ;;
  *) echo 'Unsupported GPU architecture' >&2; exit 2 ;;
esac
source "$GRALE_REPO_DIR/cluster/jean_zay/env.sh" "${GRALE_MATCHER_ARCH:-a100}"
export PYTHONPATH="$GRALE_SOLVER_DEPS:$GRALE_REPO_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export POT_BACKEND_DISABLE_PYTORCH=1 POT_BACKEND_DISABLE_JAX=1
export POT_BACKEND_DISABLE_CUPY=1 POT_BACKEND_DISABLE_TENSORFLOW=1
python - <<'PY'
import hashlib, importlib.metadata, json, os
from pathlib import Path
import ot
root = Path(os.environ['GRALE_SOLVER_DEPS']).resolve()
assert Path(os.environ['SCRATCH']).resolve() in root.parents
assert root in Path(ot.__file__).resolve().parents
assert importlib.metadata.version('POT') == '0.9.7'
record = json.loads((root / 'PACKAGE_PROVENANCE.json').read_text())
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
           for name, sha in record['files_sha256'].items())
print('ISOLATED_POT_VERIFIED', ot.__file__, flush=True)
PY
python cluster/jean_zay/check_environment.py --gpu \
  --output "$GRALE_LOG_DIR/grale-matcher-environment-$SLURM_JOB_ID.json"
srun "$GRALE_VENV/bin/python" -u scripts/run_geometric_matcher_pilot.py \
  --reference-run "$GRALE_MATCHER_REFERENCE" --inspection-targets "$GRALE_MATCHER_INSPECTION" \
  --output "$GRALE_MATCHER_OUTPUT" --device cuda --steps 5000
