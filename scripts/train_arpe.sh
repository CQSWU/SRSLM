#!/usr/bin/env bash
set -euo pipefail

project=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python_bin=${PYTHON_BIN:-$project/.venv/bin/python}
cd "$project"
exec "$python_bin" train.py \
  --config_path learning/train_arpe.yaml
