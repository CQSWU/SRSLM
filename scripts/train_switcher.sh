#!/usr/bin/env bash
set -euo pipefail

project=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python_bin=${PYTHON_BIN:-$project/.venv/bin/python}
cd "$project"
exec "$python_bin" train_switcher.py --mode final \
  --config_path learning/train_switcher.yaml
