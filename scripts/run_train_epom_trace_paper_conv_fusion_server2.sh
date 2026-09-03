#!/usr/bin/env bash
set -euo pipefail

root="${PROJECT_ROOT:-/root/epom-direct-eval}"
stage="${1:-pipeline}"
device="${CUDA_VISIBLE_DEVICES:-0}"

smoke_config="learning/train_epom_trace_paper_conv_fusion_r5_smoke.yaml"
formal_config="learning/train_epom_trace_paper_conv_fusion_r5_500m.yaml"
smoke_run="EPOM-TracePaperConvDirectCorrection-R5-Smoke-S0-20260902"
formal_run="EPOM-TracePaperConvDirectCorrection-R5-S0-20260902"
smoke_dir="$root/weights/EPOM-TracePaperConvDirectCorrection-R5-smoke"
formal_dir="$root/weights/EPOM-TracePaperConvDirectCorrection-R5-500m"
log_root="$root/logs/epom_trace_paper_conv_direct_correction_r5_20260902"
base_checkpoint="$root/weights/EPOM-lifelong-finetune-r5/EPOM-Lifelong-Finetune-R5/checkpoint_p0/checkpoint_000024418_100016128.pth"
base_sha256="f70a305ee68546be95e0a93d7f61c9aec435a50da20624a3b382af2276ad79d2"

case "$stage" in
  smoke|formal|pipeline|status) ;;
  *) echo "Usage: $0 smoke|formal|pipeline|status" >&2; exit 2 ;;
esac

if [[ -f /opt/PPU_SDK/envsetup.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/PPU_SDK/envsetup.sh >/dev/null
  set -u
elif [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/PPU_SDK/envsetup.sh >/dev/null
  set -u
fi

python_bin="$root/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
[[ -f "$base_checkpoint" ]] || { echo "Missing frozen EPOM-L checkpoint" >&2; exit 2; }
actual_base_sha256="$(sha256sum "$base_checkpoint" | awk '{print $1}')"
[[ "$actual_base_sha256" == "$base_sha256" ]] || {
  echo "Frozen EPOM-L checkpoint hash mismatch: $actual_base_sha256" >&2
  exit 3
}

mkdir -p "$log_root"
export CUDA_VISIBLE_DEVICES="$device"
export RTC_CACHE_ENABLE=1
export RTC_CACHE_PATH="$root/tmp/rtccache_epom_trace_paper_conv_direct_r5"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$RTC_CACHE_PATH"

latest_frames() {
  local checkpoint_dir="$1" latest
  latest="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'checkpoint_*.pth' \
    -printf '%f\n' 2>/dev/null | sort -t_ -k3,3n | tail -1)"
  [[ -n "$latest" ]] || return 1
  latest="${latest%.pth}"
  printf '%s\n' "${latest##*_}"
}

verify_log() {
  local log="$1"
  if grep -Eqi \
    'Traceback \(most recent call last\):|unhandled exception|RuntimeError:|ValueError:|OutOfMemoryError|CUDA out of memory|Segmentation fault' \
    "$log"; then
    echo "Training log contains an error: $log" >&2
    return 1
  fi
}

verify_run() {
  local run_dir="$1" log="$2" minimum_frames="$3"
  local frames architecture tau_raw workers
  verify_log "$log"
  frames="$(latest_frames "$run_dir/checkpoint_p0")"
  (( frames >= minimum_frames )) || {
    echo "Checkpoint shortfall: $frames < $minimum_frames" >&2
    return 1
  }
  read -r architecture tau_raw workers < <(
    "$python_bin" - "$run_dir/config.json" <<'PY'
import json
import sys
full = json.load(open(sys.argv[1], encoding="utf-8"))["full_config"]
print(
    full["experiment_settings"]["trace_context_architecture"],
    str(full["environment"]["tau_raw"]).lower(),
    full["async_ppo"]["num_workers"],
)
PY
  )
  [[ "$architecture" == "paper_entropy_fusion" ]] || return 1
  [[ "$tau_raw" == "false" ]] || return 1
  [[ "$workers" == "12" ]] || return 1
  printf 'VERIFIED run=%s frames=%s architecture=%s tau_raw=%s workers=%s\n' \
    "$run_dir" "$frames" "$architecture" "$tau_raw" "$workers"
}

run_stage() {
  local label="$1" config="$2" run="$3" train_dir="$4" target="$5"
  local log_dir log run_dir
  log_dir="$log_root/$label"
  log="$log_dir/train.log"
  run_dir="$train_dir/$run"
  mkdir -p "$log_dir"
  if pgrep -af "[t]rain.py.*--run_name $run" >/dev/null; then
    echo "$run is already running" >&2
    return 4
  fi
  {
    echo "RUN_BEGIN label=$label run=$run target=$target device=$device"
    echo "BASE_CHECKPOINT_SHA256 $actual_base_sha256"
    echo "CONFIG_SHA256 $(sha256sum "$root/$config" | awk '{print $1}')"
    echo "EXPECTED_TRAINABLE_PARAMETERS 303846"
    echo "TRACE_ENCODER conv32_two_residual_blocks_fc32"
    echo "TRACE_CONTRACT full_11x11_free_cell_mean_centered_no_network_mask"
    echo "LOGIT_RULE z_prime_equals_z_minus_entropy_gated_p"
  } >"$log_dir/provenance.txt"
  "$python_bin" -u "$root/train.py" \
    --config_path "$root/$config" \
    --run_name "$run" \
    --train_dir "$train_dir" \
    --train_for_env_steps "$target" >"$log" 2>&1
  verify_run "$run_dir" "$log" "$target" | tee "$log_dir/verification.txt"
  touch "$log_dir/COMPLETE"
}

show_status() {
  for entry in \
    "smoke|$smoke_dir/$smoke_run|1000000" \
    "formal|$formal_dir/$formal_run|500000000"; do
    IFS='|' read -r label run_dir target <<<"$entry"
    frames="$(latest_frames "$run_dir/checkpoint_p0" 2>/dev/null || printf '0')"
    printf '%s frames=%s target=%s complete=%s\n' \
      "$label" "$frames" "$target" "$([[ -f "$log_root/$label/COMPLETE" ]] && echo yes || echo no)"
  done
  pgrep -af '[t]rain.py.*EPOM-TracePaperConvDirectCorrection' || true
}

cd "$root"
case "$stage" in
  smoke)
    run_stage smoke "$smoke_config" "$smoke_run" "$smoke_dir" 1000000
    ;;
  formal)
    verify_run "$smoke_dir/$smoke_run" "$log_root/smoke/train.log" 1000000 >/dev/null
    run_stage formal "$formal_config" "$formal_run" "$formal_dir" 500000000
    ;;
  pipeline)
    if [[ ! -f "$log_root/smoke/COMPLETE" ]]; then
      run_stage smoke "$smoke_config" "$smoke_run" "$smoke_dir" 1000000
    else
      verify_run "$smoke_dir/$smoke_run" "$log_root/smoke/train.log" 1000000 >/dev/null
    fi
    run_stage formal "$formal_config" "$formal_run" "$formal_dir" 500000000
    ;;
  status)
    show_status
    ;;
esac
