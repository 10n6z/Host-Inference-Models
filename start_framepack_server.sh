#!/usr/bin/env bash
# Launch the standalone FramePack I2V server in its dedicated conda env.
# FramePack pins torch 2.6/cu124 + diffusers 0.33, so it cannot share host-models.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate host-framepack

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/gpt-lab/long/outputs}"
export HF_HOME="${HF_HOME:-/gpt-lab/long/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/gpt-lab/long/hf-cache/hub}"
export FRAMEPACK_PUBLIC_BASE_URL="${FRAMEPACK_PUBLIC_BASE_URL:-http://localhost:9000}"

# Must run from the FramePack repo so `diffusers_helper` is importable.
cd /gpt-lab/long/repos/FramePack
exec uvicorn framepack_server:app --host 0.0.0.0 --port 8012
