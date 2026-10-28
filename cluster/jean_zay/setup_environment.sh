#!/usr/bin/env bash
# Login-node package installation only. No GPU allocation or training.
set -eo pipefail
export GRALE_ARCH="${1:-a100}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_dir/cluster/jean_zay/paths.sh"
module purge
module load "arch/$GRALE_ARCH"
module load "pytorch-gpu/py3/$GRALE_TORCH_VERSION"
set -u
mkdir -p "$(dirname "$GRALE_VENV")" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TMPDIR"

# Resolve symlinks as well as checking the lexical path in paths.sh.
python - <<'PY'
import os
from pathlib import Path
scratch = Path(os.environ['SCRATCH']).resolve()
venv = Path(os.environ['GRALE_VENV']).resolve()
if scratch not in venv.parents:
    raise SystemExit('GRALE_VENV resolves outside SCRATCH')
PY
if [[ ! -x "$GRALE_VENV/bin/python" ]]; then
  if [[ -e "$GRALE_VENV" ]]; then
    echo "An incomplete environment exists; choose a new GRALE_VENV: $GRALE_VENV" >&2
    exit 2
  fi
  uv_bin="${GRALE_UV_BIN:-$WORK/tools/uv/uv}"
  if [[ -x "$uv_bin" ]]; then
    "$uv_bin" venv --seed --python "$(command -v python)" --system-site-packages "$GRALE_VENV"
  else
    python -m venv --system-site-packages "$GRALE_VENV"
  fi
fi
grep -Eq '^include-system-site-packages = true$' "$GRALE_VENV/pyvenv.cfg" || {
  echo "GRALE_VENV must inherit module site packages." >&2; exit 2;
}
source "$GRALE_VENV/bin/activate"
python - <<'PY'
import os
from pathlib import Path
import sys
import torch
if torch.__version__.split('+')[0] != os.environ['GRALE_TORCH_VERSION']:
    raise SystemExit('Existing environment does not use the selected PyTorch module')
if Path(sys.prefix).resolve() in Path(torch.__file__).resolve().parents:
    raise SystemExit('Refusing to reuse an environment containing its own PyTorch installation')
PY
records="$WORK/environment-records/grale-$GRALE_ARCH"
mkdir -p "$records"

# Protect the module's numerical/CUDA stack during normal dependency resolution.
python - "$records/module-constraints.txt" <<'PY'
import importlib.metadata as md
from pathlib import Path
import sys
names = ['torch', 'torchvision', 'torchaudio', 'numpy', 'scipy']
rows = []
for name in names:
    try:
        rows.append(f'{name}=={md.version(name)}')
    except md.PackageNotFoundError:
        if name in {'torch', 'numpy', 'scipy'}:
            raise
Path(sys.argv[1]).write_text('\n'.join(rows) + '\n')
PY
python -m pip install --constraint "$records/module-constraints.txt" \
  --requirement "$repo_dir/requirements/jean-zay.txt"
python "$repo_dir/cluster/jean_zay/check_environment.py" \
  --output "$records/environment.json"
python -m pip freeze > "$records/pip-freeze.txt"
module list 2> "$records/modules.txt"
echo "Environment ready: $GRALE_VENV"
echo "Activate: source cluster/jean_zay/env.sh $GRALE_ARCH"
