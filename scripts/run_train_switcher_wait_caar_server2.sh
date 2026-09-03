#!/usr/bin/env bash
set -euo pipefail

atomic_status() {
  local path=$1 value=$2 temporary="${1}.tmp.$$"
  printf '%s\n' "$value" >"$temporary"
  mv -f -- "$temporary" "$path"
}

fail() {
  local status_file=$1 message=$2
  atomic_status "$status_file" "FAILED: $message"
  echo "$message" >&2
  exit 1
}

if [[ ${1:-} == __run ]]; then
  [[ $# -eq 9 ]] || exit 64
  mode=$2 project=$3 python_bin=$4 config_path=$5 run_name=$6
  train_dir=$7 target_steps=$8 log_dir=$9
  status_file="$log_dir/STATUS"
  completed=0
  trap '(( completed )) || atomic_status "$status_file" "FAILED: unexpected launcher exit"' EXIT

  set +u
  # shellcheck disable=SC1091
  source /opt/PPU_SDK/envsetup.sh >/dev/null
  set -u
  cd "$project"
  export PYTHONPATH="$project" CUDA_VISIBLE_DEVICES=0
  export CPPIMPORT_RELEASE_MODE=1 RTC_CACHE_ENABLE=1
  export RTC_CACHE_PATH="$project/tmp/rtccache_switcher_wait_caar_${mode}"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
  mkdir -p "$RTC_CACHE_PATH"

  set +e
  "$python_bin" -u train_switcher_wait_caar.py --config_path "$config_path"
  rc=$?
  set -e
  (( rc == 0 )) || fail "$status_file" "wait-aware training exited with code $rc"

  run_dir="$train_dir/$run_name"
  checkpoint_json=$("$python_bin" scripts/switcher_wait_caar_artifact_contract.py \
    checkpoint --weights-dir "$run_dir")
  terminal=$("$python_bin" -c 'import json,sys;print(json.load(sys.stdin)["path"])' <<<"$checkpoint_json")
  frames=$("$python_bin" -c 'import json,sys;print(json.load(sys.stdin)["env_steps"])' <<<"$checkpoint_json")
  (( frames >= target_steps )) || fail "$status_file" "terminal checkpoint stopped at $frames before $target_steps"

  sha256sum -c "$log_dir/source_before.sha256" >/dev/null || fail "$status_file" "tracked inputs changed during training"
  cp -f -- "$log_dir/source_before.sha256" "$log_dir/source_after.sha256"
  : >"$log_dir/source_hash_diff.txt"
  "$python_bin" scripts/switcher_wait_caar_artifact_contract.py postflight \
    --weights-dir "$run_dir" --log-dir "$log_dir" --project-root "$project" \
    --checkpoint "$terminal" --expected-experiment "$run_name" \
    --expected-target-frames "$target_steps" --output "$log_dir/VALIDATION.json" >/dev/null
  atomic_status "$status_file" COMPLETE
  : >"$log_dir/COMPLETE"
  completed=1
  trap - EXIT
  echo "COMPLETE mode=$mode checkpoint=$terminal frames=$frames"
  exit 0
fi

[[ $# -eq 1 ]] || { echo "Usage: $0 smoke|formal|audit" >&2; exit 2; }
mode=$1
case "$mode" in
  smoke)
    config_name=train_switcher_wait_caar_smoke_1m_server2.yaml
    run_name=SRSLM-WaitAware-CAAR-Smoke-1M
    train_dir_name=SRSLM-switcher-wait-aware-caar-smoke
    log_name=switcher_wait_aware_caar_smoke_1m
    target_steps=1000000
    ;;
  formal|audit)
    config_name=train_switcher_wait_caar_100m_server2.yaml
    run_name=SRSLM-WaitAware-CAAR-100M
    train_dir_name=SRSLM-switcher-wait-aware-caar-100m
    log_name=switcher_wait_aware_caar_100m
    target_steps=100000000
    ;;
  *) echo "Usage: $0 smoke|formal|audit" >&2; exit 2 ;;
esac

project=${PROJECT_ROOT:-/root/epom-direct-eval}
python_bin=${PYTHON_BIN:-$project/.venv/bin/python}
config_path="$project/learning/$config_name"
train_dir="$project/weights/$train_dir_name"
run_dir="$train_dir/$run_name"
log_dir="$project/logs/$log_name"
status_file="$log_dir/STATUS"
pid_file="$log_dir/train.pid"
worker_cap=${SERVER_WORKER_CAP:-12}

[[ "$worker_cap" == 12 ]] || { echo "Server2 hard worker cap must remain 12" >&2; exit 2; }
[[ -x "$python_bin" && -s "$config_path" ]] || { echo "Missing Python/config" >&2; exit 2; }
[[ -f /opt/PPU_SDK/envsetup.sh ]] || { echo "Missing PPU SDK" >&2; exit 2; }
mkdir -p "$log_dir"
cd "$project"

set +u
# shellcheck disable=SC1091
source /opt/PPU_SDK/envsetup.sh >/dev/null
set -u

validation=$("$python_bin" train_switcher_wait_caar.py --config_path "$config_path" --validate-only)
validated=$("$python_bin" -c 'import json,sys;print(str(json.load(sys.stdin).get("validated",False)).lower())' <<<"$validation")
[[ "$validated" == true ]] || { echo "Wait-aware config validation failed" >&2; exit 4; }
decision_scope=$("$python_bin" -c 'import json,sys;print(json.load(sys.stdin)["decision_scope"])' <<<"$validation")
wait_routing=$("$python_bin" -c 'import json,sys;print(json.load(sys.stdin)["wait_routing"])' <<<"$validation")
[[ "$decision_scope" == aoreplan_nonwait_only && "$wait_routing" == aoreplan_wait_to_caar ]] || { echo "Wait routing contract differs" >&2; exit 4; }

candidate_checkpoint=$("$python_bin" -c 'import json,sys;print(json.load(sys.stdin)["candidate_policy"]["checkpoint_path"])' <<<"$validation")
candidate_weights=$("$python_bin" -c 'import json,sys;print(json.load(sys.stdin)["candidate_policy"]["weights_path"])' <<<"$validation")
base_checkpoint=$("$python_bin" -c 'import json,sys;print(json.load(sys.stdin)["candidate_policy"]["base_checkpoint_path"])' <<<"$validation")
base_weights=$("$python_bin" -c 'import json,sys;print(json.load(sys.stdin)["candidate_policy"]["base_weights_path"])' <<<"$validation")
candidate_config="$candidate_weights/config.json"
base_config="$base_weights/config.json"
planner_binary=$(find planning -maxdepth 1 -type f -name 'planner*.so' -print | sort | head -n 1)
tracked=(
  scripts/run_train_switcher_wait_caar_server2.sh
  scripts/switcher_wait_caar_artifact_contract.py scripts/switcher_artifact_contract.py
  scripts/switcher_checkpoint_identity.py train_switcher_wait_caar.py train.py "$config_path"
  pomapf_env/switcher_caar_env.py pomapf_env/switcher_env.py pomapf_env/env.py
  pomapf_env/pomapf_config.py pomapf_env/stigmergic.py pomapf_env/wrappers.py
  agents/switcher_caar_candidate.py agents/switcher_core.py agents/epom_trace_context.py
  agents/epom_trace.py agents/caar.py agents/utils_agents.py
  learning/switcher_actor_critic.py learning/switcher_learner_patch.py
  learning/epom_trace_context_actor_critic.py learning/epom_trace_multiplier_actor_critic.py
  learning/config.py learning/encoder.py learning/grid_memory.py
  planning/aoreplan_branch.py planning/ao_replan_algo.py planning/replan_algo.py
  planning/planner.cpp "$planner_binary" maps/train.yaml
  "$candidate_config" "$candidate_checkpoint" "$base_config" "$base_checkpoint"
)
for path in "${tracked[@]}"; do [[ -s "$path" ]] || { echo "Tracked input is missing: $path" >&2; exit 2; }; done

if [[ "$mode" == audit ]]; then
  "$python_bin" scripts/switcher_wait_caar_artifact_contract.py verify \
    --weights-dir "$run_dir" --log-dir "$log_dir" --project-root "$project" \
    --validation "$log_dir/VALIDATION.json" --expected-experiment "$run_name" \
    --expected-target-frames "$target_steps" >/dev/null
  echo "AUDITED $run_name"
  exit 0
fi
if [[ "$mode" == formal && ! -e "$project/logs/switcher_wait_aware_caar_smoke_1m/COMPLETE" ]]; then
  echo "Formal training requires the isolated 1M smoke COMPLETE marker" >&2
  exit 4
fi
if [[ -s "$pid_file" ]]; then
  old_pid=$(tr -d '[:space:]' <"$pid_file")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "wait-aware $mode already runs as PID $old_pid" >&2
    exit 3
  fi
fi
[[ ! -e "$log_dir/COMPLETE" ]] || { echo "$mode already complete"; exit 0; }

ppu_state=$(/opt/PPU_SDK/ppu-smi/bin/ppu-smi)
grep -q 'No running processes found' <<<"$ppu_state" || { echo "Server2 PPU0 is in use" >&2; exit 5; }
active_workers=$(ps -eo args= | awk '/multiprocessing[.]spawn.*spawn_main/ {n++} END {print n+0}')
(( active_workers + 12 <= worker_cap )) || { echo "Server2 worker cap exceeded" >&2; exit 5; }

if [[ -s "$log_dir/source_before.sha256" ]]; then
  sha256sum -c "$log_dir/source_before.sha256" >/dev/null || { echo "Refusing resume after inputs changed" >&2; exit 4; }
  initialization=resume_same_isolated_run
else
  if find "$run_dir/checkpoint_p0" -maxdepth 1 -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing unproven checkpoint directory" >&2
    exit 4
  fi
  sha256sum "${tracked[@]}" >"$log_dir/source_before.sha256"
  initialization=from_scratch_seed0
fi
: >"$log_dir/source_hash_diff.txt"
printf '%s\n' "$validation" >"$log_dir/PREFLIGHT.json"
cat >"$log_dir/PROTOCOL.md" <<EOF
# Wait-aware CAAR Switcher $mode
- Initialization: $initialization; target $target_steps frames; seed 0.
- Frozen branches: exact hash-pinned final CAAR and current AORePlan.
- Routing in training and inference: AORePlan wait -> CAAR; non-wait -> stochastic two-branch Switcher.
- Gradients: Actor/entropy/KL only on non-wait rows; Critic on all valid rows.
- Environment: 200 agents, block_both, restart, 512 steps, radius 5.
- Runtime: Server2 PPU0, 12 workers under the hard 12-worker cap.
- Selection: terminal latest regular checkpoint; no best or milestone selection.
EOF
printf '%q ' "$python_bin" -u train_switcher_wait_caar.py --config_path "$config_path" >"$log_dir/COMMAND.txt"
printf '\n' >>"$log_dir/COMMAND.txt"
atomic_status "$status_file" RUNNING

script_path=$(realpath -- "$0")
nohup bash "$script_path" __run "$mode" "$project" "$python_bin" \
  "$config_path" "$run_name" "$train_dir" "$target_steps" "$log_dir" \
  >>"$log_dir/train.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pid_file"
echo "STARTED mode=$mode pid=$pid target=$target_steps log=$log_dir/train.log"
