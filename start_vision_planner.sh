#!/usr/bin/env bash
set -euo pipefail

# vision-planner (self-hosted computer-vision job planner) runs as a host
# process, not a Docker container. vLLM's V1 engine spawns its EngineCore as
# a separate multiprocess subprocess that segfaults immediately after model
# weight loading when containerized on this host -- verified reproducible
# across base image swaps, --enforce-eager, shm_size, ipc:host, pid:host,
# cap_add SYS_NICE, and spawn/fork worker methods; the identical model and
# vllm/torch versions serve real completions without any issue run directly
# on the host. This mirrors the existing start_model_server.sh precedent
# (multimodal-legacy also runs as a host process, proxied into the Docker
# network via host.docker.internal).

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vllm-trial

GPU_UUID="${VISION_PLANNER_GPU_UUID:-GPU-c808af26-9750-cfe0-9b8a-b18871c6cfed}"

# tmux's server was started long before this script (it already hosts the
# flux2-server/model-server sessions) and captured its environment at that
# time -- `tmux new -d` spawns panes using the SERVER's original
# environment, not this script's live exports, so env vars must be set
# inline in the command string itself, not exported beforehand.
LOG_FILE="/tmp/vision-planner.log"
tmux new -d -s vision-planner \
  "source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate vllm-trial && \
    HF_HOME=/gpt-lab/long/models/hf \
    CUDA_VISIBLE_DEVICES=$GPU_UUID \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 --port 8126 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    > $LOG_FILE 2>&1; echo EXITED WITH CODE \$? >> $LOG_FILE; exec bash"

echo "vision-planner started in tmux session 'vision-planner' on port 8126, logging to $LOG_FILE"
