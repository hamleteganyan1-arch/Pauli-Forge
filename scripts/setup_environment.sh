#!/usr/bin/env bash
# Reproducible Linux x86_64 / Python 3.12 CPU development environment.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
cd "$project_dir"
python3 - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('The frozen CPU environment requires Python 3.12.')
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('The frozen CPU environment targets Linux x86_64; use documented portable extras otherwise.')
PY
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$project_dir/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR" results
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --require-hashes \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements/linux-cpu-py312.lock
.venv/bin/python -m pip install --disable-pip-version-check --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
import cvxpy, e3nn, numpy, pyscf, torch
print({'torch': torch.__version__, 'numpy': numpy.__version__,
       'pyscf': pyscf.__version__, 'cvxpy': cvxpy.__version__,
       'e3nn': e3nn.__version__, 'cuda_available': torch.cuda.is_available()})
PY
