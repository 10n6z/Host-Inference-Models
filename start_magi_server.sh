#!/usr/bin/env bash
# Launch the standalone MAGI-1 video server (subprocess-wraps entry.py).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate host-magi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/gpt-lab/long/outputs}"
export HF_HOME="${HF_HOME:-/gpt-lab/long/hf-cache}"
export MAGI_ROOT="${MAGI_ROOT:-/gpt-lab/long/repos/MAGI-1}"
export MAGI_PUBLIC_BASE_URL="${MAGI_PUBLIC_BASE_URL:-http://localhost:9000}"
# triton JIT compiles CUDA kernels at runtime: keep its tmp/cache off the full
# root partition, and expose a libcuda.so dev symlink for -lcuda linking.
export TMPDIR="${TMPDIR:-/gpt-lab/long/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gpt-lab/long/tmp/triton}"
export LIBRARY_PATH="/gpt-lab/long/lib:${LIBRARY_PATH:-}"

cd /home/long/Host-Inference-Models/servers
exec uvicorn magi_server:app --host 0.0.0.0 --port 8016
