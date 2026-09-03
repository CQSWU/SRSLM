#!/usr/bin/env bash
set -euo pipefail

project=${PROJECT_ROOT:-/root/server1}
python_bin=${PYTHON_BIN:-$project/.venv/bin/python}
output_dir=${1:-$project/results/srslm_wait_aware_caar_100m_exact960_20260903}
workers=${WORKERS:-12}
worker_cap=${SERVER_WORKER_CAP:-30}
device=${DEVICE:-0}

map_list=$project/maps/eval_capacity_intersection_n600.yaml
map_registry=$project/maps/eval.yaml
switcher_weights=$project/weights/SRSLM-switcher-wait-aware-caar-100m/SRSLM-WaitAware-CAAR-100M
training_log=$project/logs/switcher_wait_aware_caar_100m
training_validation=$training_log/VALIDATION.json
readiness_result=$project/results/srslm_wait_aware_caar_100m_smoke_v2_20260903
candidate_manifest=$project/artifacts/caar_final_candidate.json
caar_weights=$project/weights/EPOM-TracePaperConvDirectCorrection-R5-500m/EPOM-TracePaperConvDirectCorrection-R5-S0-20260902
switcher_config=$switcher_weights/config.json
switcher_checkpoint=$switcher_weights/checkpoint_p0/checkpoint_000024418_100016128.pth
caar_config=$caar_weights/config.json
caar_checkpoint=$caar_weights/checkpoint_p0/checkpoint_000122074_500015104.pth
base_weights=$project/weights/EPOM-lifelong-finetune-r5/EPOM-Lifelong-Finetune-R5
base_config=$base_weights/config.json
base_checkpoint=$base_weights/checkpoint_p0/checkpoint_000024418_100016128.pth

expected_map_sha256=da5c3d4cbd4cbdc8ce9f6b271ca258d4e7b69d6aa76524c6d17201718efb02f0
expected_map_registry_sha256=b5dcfc164dde07f0d0bd39399f1c83728c7c65bc6de73c62692142477b13de33
expected_switcher_checkpoint_sha256=4973fa420a093e043d2aafb2340863a2be3ad7dda3362ef278a98ef8c1a75185
expected_switcher_model_sha256=c2bd85a0cbcffe49dec8a393e84f022efe9bc8ce916190b497d0571acbb75aa9
expected_switcher_config_sha256=de387d7b00f7cb0d56b11d78389d702d301a39fb33a7f3f666189c685e7c0bc6
expected_training_validation_sha256=b326895585ba64ef7842003dde9cd2edf26824a4b578cdf451dd57e4a01f26f1
expected_candidate_manifest_sha256=75df038934fd10a71ce5b7e97aca7456546a18940553aa49eb454c89510e654f
expected_caar_checkpoint_sha256=497118e3aa4fbaecde35e53f31fe3126e11c1a1e5b0b621b89ac0d340002d41b
expected_caar_config_sha256=e76a2b238f196752ec358ce8946eb353caa3a4fe3e4df2a92cf812506d008747
expected_base_checkpoint_sha256=f70a305ee68546be95e0a93d7f61c9aec435a50da20624a3b382af2276ad79d2
expected_base_config_sha256=74c5cc0f1c5fdc0043bfcaa2e48e3be9c46c2c652f489a2b83379788e5da69b9

project=$(realpath -- "$project")
output_dir=$(realpath -m -- "$output_dir")
case "$output_dir" in
  "$project"/results/*) ;;
  *) echo "Output must be below $project/results" >&2; exit 2 ;;
esac
[[ "$workers" == 12 ]] || { echo "This exact960 run requires exactly 12 workers" >&2; exit 2; }
[[ "$worker_cap" == 30 ]] || { echo "Server1 worker cap must remain 30" >&2; exit 2; }
[[ "$device" == 0 || "$device" == 1 ]] || { echo "DEVICE must be physical PPU 0 or 1" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }

declare -A expected_hashes=(
  ["$map_list"]=$expected_map_sha256
  ["$map_registry"]=$expected_map_registry_sha256
  ["$switcher_config"]=$expected_switcher_config_sha256
  ["$switcher_checkpoint"]=$expected_switcher_checkpoint_sha256
  ["$training_validation"]=$expected_training_validation_sha256
  ["$candidate_manifest"]=$expected_candidate_manifest_sha256
  ["$caar_config"]=$expected_caar_config_sha256
  ["$caar_checkpoint"]=$expected_caar_checkpoint_sha256
  ["$base_config"]=$expected_base_config_sha256
  ["$base_checkpoint"]=$expected_base_checkpoint_sha256
)
for path in "${!expected_hashes[@]}"; do
  [[ -s "$path" ]] || { echo "Missing frozen artifact: $path" >&2; exit 2; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "${expected_hashes[$path]}" ]] || {
    echo "Frozen artifact SHA256 differs: $path" >&2
    exit 4
  }
done

checkpoint_json=$(
  "$python_bin" scripts/switcher_artifact_contract.py checkpoint \
    --weights-dir "$switcher_weights"
)
"$python_bin" - "$checkpoint_json" "$expected_switcher_checkpoint_sha256" \
  "$expected_switcher_model_sha256" <<'PY'
import json
import sys
checkpoint = json.loads(sys.argv[1])
if checkpoint.get("env_steps") != 100_016_128:
    raise SystemExit("Terminal Switcher frame count differs")
if checkpoint.get("checkpoint_sha256") != sys.argv[2]:
    raise SystemExit("Terminal Switcher checkpoint SHA256 differs")
if checkpoint.get("policy_model_sha256") != sys.argv[3]:
    raise SystemExit("Terminal Switcher policy-model SHA256 differs")
PY
"$python_bin" scripts/validate_switcher_wait_caar_readiness.py \
  --project-root "$project" \
  --result-dir "$readiness_result" \
  --weights-dir "$switcher_weights" \
  --training-validation "$training_validation" \
  --output "$readiness_result/VALIDATION.json" >/dev/null

if [[ -s /usr/local/PPU_SDK/envsetup.sh ]]; then
  ppu_sdk_root=/usr/local/PPU_SDK
elif [[ -s /opt/PPU_SDK/envsetup.sh ]]; then
  ppu_sdk_root=/opt/PPU_SDK
else
  echo "PPU SDK is missing" >&2
  exit 2
fi
set +u
# shellcheck disable=SC1090
source "$ppu_sdk_root/envsetup.sh" >/dev/null
set -u
ppu_smi=$ppu_sdk_root/ppu-smi/bin/ppu-smi
[[ -x "$ppu_smi" ]] || { echo "ppu-smi is missing" >&2; exit 2; }

mkdir -p "$project/tmp"
exec 8>"$project/tmp/srslm_wait_aware_caar_100m_exact960_ppu${device}.lock"
flock -n 8 || { echo "Another exact960 task owns physical PPU$device" >&2; exit 5; }
target_processes=$(
  "$ppu_smi" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits -i "$device"
)
[[ -z ${target_processes//[[:space:]]/} ]] || {
  echo "Server1 physical PPU$device is already in use" >&2
  printf '%s\n' "$target_processes" >&2
  exit 5
}
active_workers=$(ps -eo args= | awk '/multiprocessing[.]spawn.*spawn_main/ {n++} END {print n+0}')
(( active_workers + workers <= worker_cap )) || {
  echo "Server1 worker cap exceeded: active=$active_workers requested=$workers cap=$worker_cap" >&2
  exit 5
}

cd "$project"
code_tracked=(
  scripts/run_srslm_wait_aware_caar_100m_exact960_server1.sh
  scripts/validate_srslm_wait_aware_caar_100m_exact960.py
  scripts/validate_switcher_wait_caar_readiness.py
  scripts/validate_srslm_wait_ablation_exact960.py
  scripts/switcher_wait_caar_artifact_contract.py
  scripts/switcher_artifact_contract.py
  scripts/switcher_checkpoint_identity.py
  run_experiments.py
  train.py
  agents/srslm.py
  agents/switcher.py
  agents/switcher_caar_candidate.py
  agents/epom_trace_context.py
  agents/epom_trace.py
  agents/caar.py
  agents/utils_agents.py
  agents/switcher_core.py
  agents/reverse_metrics.py
  learning/epom_trace_context_actor_critic.py
  learning/epom_trace_multiplier_actor_critic.py
  learning/switcher_actor_critic.py
  learning/switcher_learner_patch.py
  learning/config.py
  learning/encoder.py
  learning/grid_memory.py
  planning/aoreplan_branch.py
  planning/ao_replan_algo.py
  planning/replan_algo.py
  planning/planner.cpp
  pomapf_env/switcher_env.py
  pomapf_env/env.py
  pomapf_env/pomapf_config.py
  pomapf_env/stigmergic.py
  pomapf_env/wrappers.py
)
planner_binary=$(find planning -maxdepth 1 -type f -name 'planner*.so' -print | sort | head -n 1)
[[ -s "$planner_binary" ]] || { echo "Compiled planner binary is missing" >&2; exit 2; }
code_tracked+=("$planner_binary")
artifact_tracked=(
  "$map_list"
  "$map_registry"
  "$switcher_config"
  "$switcher_checkpoint"
  "$training_validation"
  "$training_log/source_before.sha256"
  "$training_log/source_after.sha256"
  "$training_log/STATUS"
  "$candidate_manifest"
  "$caar_config"
  "$caar_checkpoint"
  "$base_config"
  "$base_checkpoint"
)
for path in "${code_tracked[@]}" "${artifact_tracked[@]}"; do
  [[ -s "$path" ]] || { echo "Missing tracked input: $path" >&2; exit 2; }
done

if [[ -e "$output_dir" ]]; then
  echo "Output already exists; refusing to mix runs: $output_dir" >&2
  exit 3
fi
mkdir -p "$output_dir/logs"
sha256sum "${code_tracked[@]}" >"$output_dir/code_before.sha256"
sha256sum "${artifact_tracked[@]}" >"$output_dir/artifact_before.sha256"
cat "$output_dir/code_before.sha256" "$output_dir/artifact_before.sha256" \
  >"$output_dir/source_before.sha256"
code_snapshot_sha256=$(sha256sum "$output_dir/code_before.sha256" | awk '{print $1}')

"$python_bin" - "$output_dir/RUN_CONTRACT.json" "$code_snapshot_sha256" "$workers" <<'PY'
import json
import os
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema": "srslm_wait_aware_caar_100m_exact960_run_contract_v1",
    "algorithm": "SRSLM",
    "hybrid_mode": "aoreplan_wait_bypass_switcher_v3",
    "map_list_sha256": "da5c3d4cbd4cbdc8ce9f6b271ca258d4e7b69d6aa76524c6d17201718efb02f0",
    "populations": [100, 200, 300, 400, 500, 600],
    "seeds": [0, 42, 123, 2024, 3407],
    "collision_system": "block_both",
    "on_target": "restart",
    "max_steps": 512,
    "obs_radius": 5,
    "expected_rows": 960,
    "switcher_frames": 100016128,
    "switcher_checkpoint_sha256": "4973fa420a093e043d2aafb2340863a2be3ad7dda3362ef278a98ef8c1a75185",
    "switcher_policy_model_sha256": "c2bd85a0cbcffe49dec8a393e84f022efe9bc8ce916190b497d0571acbb75aa9",
    "caar_checkpoint_sha256": "497118e3aa4fbaecde35e53f31fe3126e11c1a1e5b0b621b89ac0d340002d41b",
    "training_validation_sha256": "b326895585ba64ef7842003dde9cd2edf26824a4b578cdf451dd57e4a01f26f1",
    "candidate_manifest_sha256": "75df038934fd10a71ce5b7e97aca7456546a18940553aa49eb454c89510e654f",
    "code_snapshot_sha256": sys.argv[2],
    "workers": int(sys.argv[3]),
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
journal_contract=$(sha256sum "$output_dir/RUN_CONTRACT.json" | awk '{print $1}')

cat >"$output_dir/PROTOCOL.md" <<EOF
# SRSLM wait-aware CAAR 100M exact960

- 32 frozen evaluation maps; populations 100, 200, 300, 400, 500, 600.
- Seeds 0, 42, 123, 2024, 3407; 960 unique episodes.
- Lifelong restart, 512 steps, radius 5, block_both.
- AORePlan wait routes directly to frozen CAAR; non-wait states use the PPO Switcher.
- Switcher terminal frame count: 100,016,128.
- Switcher checkpoint SHA256: $expected_switcher_checkpoint_sha256.
- CAAR checkpoint SHA256: $expected_caar_checkpoint_sha256.
- Physical PPU$device with $workers workers; Server1 aggregate cap $worker_cap.
EOF

atomic_status() {
  local value=$1
  local temporary="$output_dir/.STATUS.tmp.$$"
  printf '%s\n' "$value" >"$temporary"
  mv -f -- "$temporary" "$output_dir/STATUS"
}
atomic_status RUNNING
complete=0
on_exit() {
  if (( ! complete )); then
    atomic_status INCOMPLETE
  fi
}
trap on_exit EXIT INT TERM

export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$device"
export CPPIMPORT_RELEASE_MODE=1
export RTC_CACHE_ENABLE=1
export RTC_CACHE_PATH="$project/tmp/rtccache_srslm_wait_aware_caar_100m_exact960_ppu${device}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$RTC_CACHE_PATH"

command=(
  "$python_bin" -u run_experiments.py
  --algorithms SRSLM
  --switcher-weights-path "$switcher_weights"
  --caar-weights-path "$caar_weights"
  --caar-candidate-manifest "$candidate_manifest"
  --map-list "$map_list"
  --agents 100,200,300,400,500,600
  --seeds 0,42,123,2024,3407
  --obs-radius 5
  --max-steps 512
  --on-target restart
  --collision-system block_both
  --workers "$workers"
  --cache-algorithms
  --main-dir "$project"
  --output-dir "$output_dir"
  --output srslm_wait_aware_caar_100m_exact960.json
  --result-journal "$output_dir/results.journal.jsonl"
  --result-journal-contract "$journal_contract"
  --resume-result-journal
)
printf '%q ' "${command[@]}" >"$output_dir/COMMAND.txt"
printf '\n' >>"$output_dir/COMMAND.txt"
"${command[@]}" 2>&1 | tee "$output_dir/logs/evaluation.log"

sha256sum "${code_tracked[@]}" >"$output_dir/code_after.sha256"
sha256sum "${artifact_tracked[@]}" >"$output_dir/artifact_after.sha256"
cat "$output_dir/code_after.sha256" "$output_dir/artifact_after.sha256" \
  >"$output_dir/source_after.sha256"
diff -u "$output_dir/source_before.sha256" "$output_dir/source_after.sha256" \
  >"$output_dir/source_hash_diff.txt" || true

"$python_bin" scripts/validate_srslm_wait_aware_caar_100m_exact960.py \
  --project-root "$project" \
  --input "$output_dir/srslm_wait_aware_caar_100m_exact960.json" \
  --map-list "$map_list" \
  --output-dir "$output_dir" \
  --weights-dir "$switcher_weights" \
  --training-validation "$training_validation" \
  --candidate-manifest "$candidate_manifest" \
  --result-journal "$output_dir/results.journal.jsonl" \
  --expected-journal-contract-sha256 "$journal_contract" \
  --expected-code-snapshot-sha256 "$code_snapshot_sha256" \
  --expected-workers "$workers" \
  2>&1 | tee "$output_dir/logs/validation.log"

complete=1
trap - EXIT INT TERM
echo "COMPLETE $output_dir"
