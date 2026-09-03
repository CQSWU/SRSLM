#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/epom-direct-eval}"
EPOM_FINETUNE_MODE="${EPOM_FINETUNE_MODE:-formal}"
EPOM_FINETUNE_DEVICE="${EPOM_FINETUNE_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

case "${EPOM_FINETUNE_MODE}" in
  smoke)
    CONFIG_REL="learning/train_epom_lifelong_finetune_r5_smoke.yaml"
    RUN_NAME="EPOM-Lifelong-Finetune-R5-Smoke"
    TARGET_STEPS=1048576
    ;;
  formal)
    CONFIG_REL="learning/train_epom_lifelong_finetune_r5_100m.yaml"
    RUN_NAME="EPOM-Lifelong-Finetune-R5"
    TARGET_STEPS="${EPOM_FINETUNE_STEPS:-100000000}"
    if [[ "${TARGET_STEPS}" != "100000000" && "${TARGET_STEPS}" != "250000000" ]]; then
      echo "Formal target must be 100000000 or the approved extension 250000000" >&2
      exit 2
    fi
    ;;
  *)
    echo "EPOM_FINETUNE_MODE must be smoke or formal" >&2
    exit 2
    ;;
esac

CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_REL}"
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_NAME}"
PID_FILE="${LOG_DIR}/train.pid"
TRAIN_LOG="${LOG_DIR}/train.log"
PREFLIGHT_JSON="${LOG_DIR}/preflight.json"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config is unavailable: ${CONFIG_PATH}" >&2
  exit 2
fi

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
else
  echo "PPU SDK environment is unavailable" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
if [[ -s "${PID_FILE}" ]]; then
  existing_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "EPOM fine-tuning is already running as PID ${existing_pid}" >&2
    exit 3
  fi
fi
if pgrep -af "[t]rain.py.*${CONFIG_REL}" >/dev/null; then
  echo "A matching EPOM fine-tuning process already exists; refusing to duplicate it" >&2
  exit 3
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/preflight_epom_lifelong_finetune.py \
  --project-root "${PROJECT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --output "${PREFLIGHT_JSON}"

export CUDA_VISIBLE_DEVICES="${EPOM_FINETUNE_DEVICE}"
export RTC_CACHE_ENABLE=1
export RTC_CACHE_PATH="${RTC_CACHE_PATH:-${PROJECT_ROOT}/tmp/rtccache_epom_finetune_r5}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "${RTC_CACHE_PATH}"

printf '%q ' "${PYTHON_BIN}" train.py --config_path "${CONFIG_PATH}" \
  --train_for_env_steps "${TARGET_STEPS}" > "${LOG_DIR}/COMMAND.txt"
printf '\n' >> "${LOG_DIR}/COMMAND.txt"

nohup "${PYTHON_BIN}" -u train.py \
  --config_path "${CONFIG_PATH}" \
  --train_for_env_steps "${TARGET_STEPS}" \
  > "${TRAIN_LOG}" 2>&1 &

pid=$!
echo "${pid}" > "${PID_FILE}"
printf 'Started %s PID=%s target=%s log=%s\n' \
  "${RUN_NAME}" "${pid}" "${TARGET_STEPS}" "${TRAIN_LOG}"
