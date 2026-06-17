#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  source /home/long/miniconda3/etc/profile.d/conda.sh
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate host-models-flux2

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/gpt-lab/long/outputs}"
export FLUX2_PUBLIC_BASE_URL="${FLUX2_PUBLIC_BASE_URL:-http://localhost:9000}"

cd "$SCRIPT_DIR/servers"
exec uvicorn flux2_server:app --host 0.0.0.0 --port 8011
