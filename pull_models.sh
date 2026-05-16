#!/usr/bin/env bash
set -e

export HF_HOME=/home/long/local-ai/models/hf-cache
export HF_HUB_CACHE=/home/long/local-ai/models/hf-cache/hub

mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"

mkdir -p /home/long/local-ai/models/text-to-image
mkdir -p /home/long/local-ai/models/text-to-speech
mkdir -p /home/long/local-ai/models/text-to-audio

echo "Pulling text-to-image models..."

hf download black-forest-labs/FLUX.1-schnell \
  --local-dir /home/long/local-ai/models/text-to-image/flux-1-schnell

hf download stabilityai/stable-diffusion-3.5-medium \
  --local-dir /home/long/local-ai/models/text-to-image/stable-diffusion-3.5-medium

hf download fal/AuraFlow-v0.3 \
  --local-dir /home/long/local-ai/models/text-to-image/auraflow-v0.3

hf download TencentARC/PhotoMaker-V2 \
  --local-dir /home/long/local-ai/models/text-to-image/photomaker-v2

hf download ostris/OpenFLUX.1 \
  --local-dir /home/long/local-ai/models/text-to-image/openflux-1

echo "Pulling text-to-speech models..."

hf download hexgrad/Kokoro-82M \
  --local-dir /home/long/local-ai/models/text-to-speech/kokoro-82m

hf download fishaudio/fish-speech-1.5 \
  --local-dir /home/long/local-ai/models/text-to-speech/fish-speech-v1.5

hf download FunAudioLLM/CosyVoice2-0.5B \
  --local-dir /home/long/local-ai/models/text-to-speech/cosyvoice2-0.5b

hf download IndexTeam/IndexTTS-2 \
  --local-dir /home/long/local-ai/models/text-to-speech/indextts-2

echo "Pulling text-to-audio models..."

hf download stabilityai/stable-audio-open-1.0 \
  --local-dir /home/long/local-ai/models/text-to-audio/stable-audio-open-1.0

echo "All selected models pulled successfully."
