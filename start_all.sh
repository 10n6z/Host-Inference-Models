#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpt-lab/long/outputs}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-http://localhost:9000}"
GATEWAY_PUBLIC_BASE_URL="${GATEWAY_PUBLIC_BASE_URL:-http://localhost:9000}"
GATEWAY_TIMEOUT_SECONDS="${GATEWAY_TIMEOUT_SECONDS:-3600}"
INFERENCE_TIMEOUT_SECONDS="${INFERENCE_TIMEOUT_SECONDS:-3600}"
CUDA_VISIBLE_DEVICES_DEFAULT="${CUDA_VISIBLE_DEVICES_DEFAULT:-0,1,2,3}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$1 not found" >&2
    exit 1
  fi
}

load_conda() {
  local conda_sh="${CONDA_SH:-}"
  if [[ -z "$conda_sh" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "/gpt-lab/long/conda/etc/profile.d/conda.sh" ]]; then
      conda_sh="/gpt-lab/long/conda/etc/profile.d/conda.sh"
    else
      conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
    fi
  fi
  # shellcheck disable=SC1090
  source "$conda_sh"
}

start_tmux() {
  local session="$1"
  local command="$2"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "$session already running"
    return 0
  fi
  tmux new-session -d -s "$session" "bash -lc "
  echo "started $session"
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-60}"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name ready: $url"
      return 0
    fi
    sleep 2
  done
  echo "$name not ready after $((attempts * 2))s: $url" >&2
  return 1
}

require_cmd docker
require_cmd tmux
require_cmd curl
load_conda

export OUTPUT_ROOT PUBLIC_BASE_URL GATEWAY_PUBLIC_BASE_URL GATEWAY_TIMEOUT_SECONDS INFERENCE_TIMEOUT_SECONDS
export OUTPUTS_HOST_DIR="$OUTPUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

cd "$SCRIPT_DIR"
docker compose up -d

start_tmux "model-gateway-legacy" "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate host-models && cd \"$SCRIPT_DIR/servers\" && export OUTPUT_ROOT=\"$OUTPUT_ROOT\" PUBLIC_BASE_URL=\"$PUBLIC_BASE_URL\" INFERENCE_TIMEOUT_SECONDS=\"$INFERENCE_TIMEOUT_SECONDS\" CUDA_VISIBLE_DEVICES=\"${MODEL_SERVER_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES_DEFAULT}\" PYTORCH_CUDA_ALLOC_CONF=\"$PYTORCH_CUDA_ALLOC_CONF\" CUDA_MODULE_LOADING=\"$CUDA_MODULE_LOADING\" && exec uvicorn model_server:app --host 0.0.0.0 --port 8001 2>&1 | tee -a \"$LOG_DIR/model-server.log\""

start_tmux "flux2-server" "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate host-models-flux2 && cd \"$SCRIPT_DIR/servers\" && export OUTPUT_ROOT=\"$OUTPUT_ROOT\" FLUX2_PUBLIC_BASE_URL=\"$PUBLIC_BASE_URL\" INFERENCE_TIMEOUT_SECONDS=\"$INFERENCE_TIMEOUT_SECONDS\" CUDA_VISIBLE_DEVICES=\"${FLUX2_CUDA_VISIBLE_DEVICES:-0}\" PYTORCH_CUDA_ALLOC_CONF=\"$PYTORCH_CUDA_ALLOC_CONF\" CUDA_MODULE_LOADING=\"$CUDA_MODULE_LOADING\" && exec uvicorn flux2_server:app --host 0.0.0.0 --port 8011 2>&1 | tee -a \"$LOG_DIR/flux2-server.log\""

start_tmux "magi-server" "source \"$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate host-magi && cd \"$SCRIPT_DIR/servers\" && export OUTPUT_ROOT=\"$OUTPUT_ROOT\" HF_HOME=\"${HF_HOME:-/gpt-lab/long/hf-cache}\" MAGI_ROOT=\"${MAGI_ROOT:-/gpt-lab/long/repos/MAGI-1}\" MAGI_PUBLIC_BASE_URL=\"$PUBLIC_BASE_URL\" TMPDIR=\"${TMPDIR:-/gpt-lab/long/tmp}\" TRITON_CACHE_DIR=\"${TRITON_CACHE_DIR:-/gpt-lab/long/tmp/triton}\" LIBRARY_PATH=\"/gpt-lab/long/lib:\${LIBRARY_PATH:-}\" CUDA_VISIBLE_DEVICES=\"${MAGI_CUDA_VISIBLE_DEVICES:-0}\" && exec uvicorn magi_server:app --host 0.0.0.0 --port 8016 2>&1 | tee -a \"$LOG_DIR/magi-server.log\""

if [[ -d "${FRAMEPACK_ROOT:-/gpt-lab/long/repos/FramePack}" ]]; then
  start_tmux "framepack-server" "source \"\/Users/bfn799/miniconda3/etc/profile.d/conda.sh\" && conda activate /gpt-lab/long/conda/envs/host-framepack && cd \"${FRAMEPACK_ROOT:-/gpt-lab/long/repos/FramePack}\" && export OUTPUT_ROOT=\"$OUTPUT_ROOT\" HF_HOME=\"${HF_HOME:-/gpt-lab/long/hf-cache}\" HF_HUB_CACHE=\"${HF_HUB_CACHE:-/gpt-lab/long/hf-cache/hub}\" FRAMEPACK_PUBLIC_BASE_URL=\"$PUBLIC_BASE_URL\" CUDA_VISIBLE_DEVICES=\"${FRAMEPACK_CUDA_VISIBLE_DEVICES:-2}\" && exec uvicorn framepack_server:app --host 0.0.0.0 --port 8012 2>&1 | tee -a \"$LOG_DIR/framepack-server.log\""
else
  echo "FramePack root missing; skipped framepack-server"
fi

wait_http "gateway" "http://localhost:9000/health" 90 || wait_http "gateway-models" "http://localhost:9000/models" 90
wait_http "model-server" "http://localhost:8001/health" 90
wait_http "flux2" "http://localhost:8011/health" 90
wait_http "magi" "http://localhost:8016/health" 90 || true
wait_http "framepack" "http://localhost:8012/health" 90 || true

docker compose ps
tmux list-sessions | grep -E "model-gateway-legacy|flux2-server|magi-server|framepack-server" || true
