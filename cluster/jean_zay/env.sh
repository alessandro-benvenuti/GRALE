#!/usr/bin/env bash
# Usage: source cluster/jean_zay/env.sh a100 [cpu]
export GRALE_ARCH="${1:-${GRALE_ARCH:-a100}}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh" || return
module purge || return
if [[ "${2:-gpu}" != cpu ]]; then
  module load "arch/$GRALE_ARCH" || return
fi
module load "pytorch-gpu/py3/$GRALE_TORCH_VERSION" || return
if [[ ! -x "$GRALE_VENV/bin/python" ]]; then
  echo "Missing environment: $GRALE_VENV" >&2
  echo "Run: bash cluster/jean_zay/setup_environment.sh $GRALE_ARCH" >&2
  return 2
fi
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$MPLCONFIGDIR" \
  "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TMPDIR"
source "$GRALE_VENV/bin/activate"
python - <<'PY'
import os
from pathlib import Path
import sys
if Path(os.environ['SCRATCH']).resolve() not in Path(sys.prefix).resolve().parents:
    raise SystemExit('Active environment resolves outside SCRATCH')
PY
if [[ $? != 0 ]]; then return 2; fi
cd "$GRALE_REPO_DIR" || return
