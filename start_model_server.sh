#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate host-models

export OUTPUT_ROOT=/gpt-lab/long/outputs 
export PUBLIC_BASE_URL=http://localhost:9000 
export DEVICE=cuda
export IMAGE_GPU_UUID="GPU-abffce46-a266-a0b8-8f86-e40d19fd546e"
export VIDEO_GPU_UUID="GPU-c808af26-9750-cfe0-9b8a-b18871c6cfed"
export CUDA_VISIBLE_DEVICES="${MODEL_SERVER_CUDA_VISIBLE_DEVICES:-${IMAGE_GPU_UUID},${VIDEO_GPU_UUID}}"

cd "$SCRIPT_DIR"
docker compose up -d
docker compose ps

cd "$SCRIPT_DIR/servers"
tmux new -d -s model-server "exec uvicorn model_server:app --host 0.0.0.0 --port 8001"
