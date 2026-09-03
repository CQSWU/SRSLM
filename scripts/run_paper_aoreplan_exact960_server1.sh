#!/usr/bin/env bash
set -euo pipefail

project=${PROJECT_ROOT:-/root/server1}
python_bin=${PYTHON_BIN:-$project/.venv/bin/python}
map_list=${MAP_LIST:-$project/maps/eval_capacity_intersection_n600.yaml}
output_dir=${1:-$project/results/paper_aoreplan_exact960_20260829}
workers=30
worker_cap=${SERVER_WORKER_CAP:-30}
expected_map_sha256=da5c3d4cbd4cbdc8ce9f6b271ca258d4e7b69d6aa76524c6d17201718efb02f0

project=$(realpath -- "$project")
output_dir=$(realpath -m -- "$output_dir")
map_list=$(realpath -- "$map_list")
case "$output_dir" in
  "$project"/results/*) ;;
  *) echo "Output must be a new directory below $project/results" >&2; exit 2 ;;
esac
[[ "$worker_cap" == "30" ]] || {
  echo "Server1 hard worker cap must remain 30" >&2
  exit 2
}
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
[[ -s "$map_list" ]] || { echo "Missing fixed map list: $map_list" >&2; exit 2; }
[[ "$(sha256sum "$map_list" | awk '{print $1}')" == "$expected_map_sha256" ]] || {
  echo "Fixed map-list SHA256 differs" >&2
  exit 4
}
[[ ! -e "$output_dir" ]] || {
  echo "Refusing to reuse any existing output path: $output_dir" >&2
  exit 3
}

active_workers=$(ps -eo args= | awk '/multiprocessing[.]spawn.*spawn_main/ {n++} END {print n+0}')
(( active_workers + workers <= worker_cap )) || {
  echo "Server1 worker cap would be exceeded: active=$active_workers requested=$workers cap=$worker_cap" >&2
  exit 5
}

cd "$project"
planner_binary=$(find planning -maxdepth 1 -type f -name 'planner*.so' -print | sort | head -n 1)
[[ -s "$planner_binary" ]] || { echo "Missing compiled planner binary" >&2; exit 2; }

tracked=(
  scripts/run_paper_aoreplan_exact960_server1.sh
  scripts/validate_paper_exact960.py
  run_experiments.py
  agents/ao_replan.py
  agents/replan.py
  agents/reverse_metrics.py
  planning/ao_replan_algo.py
  planning/aoreplan_branch.py
  planning/replan_algo.py
  planning/planner.cpp
  "$planner_binary"
  pomapf_env/env.py
  pomapf_env/wrappers.py
  pomapf_env/pomapf_config.py
  "$map_list"
)
for path in "${tracked[@]}"; do
  [[ -s "$path" ]] || { echo "Tracked input is missing: $path" >&2; exit 2; }
done

mkdir -p "$output_dir/logs"
printf 'RUNNING\n' >"$output_dir/STATUS"
completed=0
mark_failure() {
  if (( ! completed )); then
    printf 'FAILED\n' >"$output_dir/STATUS"
  fi
}
trap mark_failure EXIT

sha256sum "${tracked[@]}" >"$output_dir/source_before.sha256"
cat >"$output_dir/PROTOCOL.md" <<EOF
# AORePlan lifelong exact960

- Algorithm label: AORePlan.
- Maps: the fixed 32-map capacity intersection (SHA256 $expected_map_sha256).
- Populations: 100, 200, 300, 400, 500, 600.
- Seeds: 0, 42, 123, 2024, 3407.
- Environment: lifelong restart, 512 steps, radius 5, block_both collisions.
- Reverse means returning to the immediately previous timestep position; waits and blocked steps are recorded.
- Dynamic no-path fallback follows RePlan: 50% wait and 50% random statically free direction.
- Static A* is queried only for a reverse movement proposal.
- Exactly 30 CPU workers; CUDA is hidden.
- Pre-existing multiprocessing workers: $active_workers; Server1 hard cap: $worker_cap.
EOF

export PYTHONPATH="$project"
export CUDA_VISIBLE_DEVICES=
export CPPIMPORT_RELEASE_MODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

printf '%q ' "$python_bin" -u run_experiments.py \
  --algorithms AORePlan --map-list "$map_list" \
  --agents 100,200,300,400,500,600 --seeds 0,42,123,2024,3407 \
  --obs-radius 5 --max-steps 512 --on-target restart \
  --collision-system block_both --workers "$workers" --cache-algorithms \
  --main-dir "$project" --output-dir "$output_dir" --output aoreplan_exact960.json \
  >"$output_dir/COMMAND.txt"
printf '\n' >>"$output_dir/COMMAND.txt"

"$python_bin" -u run_experiments.py \
  --algorithms AORePlan \
  --map-list "$map_list" \
  --agents 100,200,300,400,500,600 \
  --seeds 0,42,123,2024,3407 \
  --obs-radius 5 \
  --max-steps 512 \
  --on-target restart \
  --collision-system block_both \
  --workers "$workers" \
  --cache-algorithms \
  --main-dir "$project" \
  --output-dir "$output_dir" \
  --output aoreplan_exact960.json \
  2>&1 | tee "$output_dir/logs/aoreplan_exact960.log"

sha256sum "${tracked[@]}" >"$output_dir/source_after.sha256"
diff -u "$output_dir/source_before.sha256" "$output_dir/source_after.sha256" \
  >"$output_dir/source_hash_diff.txt" || true

"$python_bin" scripts/validate_paper_exact960.py \
  --algorithm AORePlan \
  --input "$output_dir/aoreplan_exact960.json" \
  --map-list "$map_list" \
  --output-dir "$output_dir" \
  --expected-workers "$workers" \
  2>&1 | tee "$output_dir/logs/validation.log"

completed=1
trap - EXIT
echo "COMPLETE $output_dir"
