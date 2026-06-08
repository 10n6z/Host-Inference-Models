#!/usr/bin/env bash
set -e

export HF_HOME=/gpt-lab/long/models/hf-cache
export HF_HUB_CACHE=/gpt-lab/long/models/hf-cache/hub

mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"

mkdir -p /gpt-lab/long/models/text-to-video

echo "Pulling text-to-video models..."

hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir /gpt-lab/long/models/text-to-video/wan2.1-t2v-1.3b

hf download Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --local-dir /gpt-lab/long/models/text-to-video/wan2.1-t2v-14b

hf download Lightricks/LTX-Video \
  --local-dir /gpt-lab/long/models/text-to-video/ltx-video

hf download THUDM/CogVideoX-2b \
  --local-dir /gpt-lab/long/models/text-to-video/cogvideox-2b

hf download THUDM/CogVideoX-5b \
  --local-dir /gpt-lab/long/models/text-to-video/cogvideox-5b

hf download genmo/mochi-1-preview \
  --local-dir /gpt-lab/long/models/text-to-video/mochi-1-preview

hf download hunyuanvideo-community/HunyuanVideo \
  --local-dir /gpt-lab/long/models/text-to-video/hunyuanvideo

echo "All video models pulled successfully."
