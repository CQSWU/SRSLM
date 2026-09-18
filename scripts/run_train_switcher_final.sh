#!/usr/bin/env bash
set -euo pipefail

project=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python_bin=${PYTHON_BIN:-$project/.venv/bin/python}
cd "$project"
exec "$python_bin" train_switcher_wait_arpe.py \
  --config_path learning/train_switcher_final_1b.yaml
