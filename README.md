# Host-Inference-Models

Local GPU inference services for image generation and audio generation.

## Services

- Image service: `servers/image_server.py` on port `8001`
- Audio/TTS service: `servers/audio_server.py` on port `8002`

Both services read `servers/.env`, use same model/cache roots, and mount `OUTPUT_ROOT` at `/outputs`.

Generated files:

- Images: `outputs/images`
- Audio: `outputs/audio`

## Environment

Copy `servers/.env.example` to `servers/.env` and edit paths:

```bash
MODEL_ROOT=/home/long/local-ai/models
HF_HOME=/home/long/local-ai/models/hf-cache
HF_HUB_CACHE=/home/long/local-ai/models/hf-cache/hub
OUTPUT_ROOT=/home/long/local-ai/outputs

IMAGE_PUBLIC_BASE_URL=http://YOUR_GPU_NODE_IP:8001
AUDIO_PUBLIC_BASE_URL=http://YOUR_GPU_NODE_IP:8002

FLUX_MODEL_PATH=/home/long/local-ai/models/text-to-image/flux-1-schnell
SD35_MODEL_PATH=/home/long/local-ai/models/text-to-image/stable-diffusion-3.5-medium
AURAFLOW_MODEL_PATH=/home/long/local-ai/models/text-to-image/auraflow-v0.3
OPENFLUX_MODEL_PATH=/home/long/local-ai/models/text-to-image/openflux-1
STABLE_AUDIO_MODEL_PATH=/home/long/local-ai/models/text-to-audio/stable-audio-open-1.0
COSYVOICE2_MODEL_PATH=/home/long/local-ai/models/text-to-speech/cosyvoice2-0.5b
FISH_SPEECH_MODEL_PATH=/home/long/local-ai/models/text-to-speech/fish-speech-v1.5
INDEXTTS2_MODEL_PATH=/home/long/local-ai/models/text-to-speech/indextts-2

# Optional fallback references for models that need prompt audio:
COSYVOICE2_DEFAULT_REFERENCE_AUDIO=
COSYVOICE2_DEFAULT_REFERENCE_TEXT=
INDEXTTS2_DEFAULT_REFERENCE_AUDIO=
```

## Download Models

Do not commit model weights, generated outputs, or Hugging Face cache files.

```bash
mkdir -p /home/long/local-ai/models/text-to-speech

hf download FunAudioLLM/CosyVoice2-0.5B \
  --local-dir /home/long/local-ai/models/text-to-speech/cosyvoice2-0.5b

hf download fishaudio/fish-speech-1.5 \
  --local-dir /home/long/local-ai/models/text-to-speech/fish-speech-v1.5

hf download IndexTeam/IndexTTS-2 \
  --local-dir /home/long/local-ai/models/text-to-speech/indextts-2
```

CosyVoice2, Fish Speech, and IndexTTS-2 use custom runners and may need model-specific dependencies. Add dependencies one model at a time. If package versions conflict with working image/audio services, isolate them into separate environments or services instead of changing the shared environment blindly.

## Start Image Server

```bash
chmod +x start_image_server.sh
./start_image_server.sh
```

Runs:

```bash
uvicorn image_server:app --host 0.0.0.0 --port 8001
```

## Start Audio Server

```bash
chmod +x start_audio_server.sh
./start_audio_server.sh
```

Runs:

```bash
uvicorn audio_server:app --host 0.0.0.0 --port 8002
```

## Run Both With tmux

```bash
tmux new -d -s local-ai-image "/home/long/local-ai/start_image_server.sh"
tmux new -d -s local-ai-audio "/home/long/local-ai/start_audio_server.sh"
tmux ls
tmux attach -t local-ai-image
tmux attach -t local-ai-audio
```

## Smoke Tests

Health:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8002/models
```

FLUX image:

```bash
curl -X POST http://localhost:8001/generate/image/flux \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cinematic mountain lake at sunrise",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance_scale": 0
  }'
```

Kokoro TTS:

```bash
curl -X POST http://localhost:8002/generate/tts/kokoro \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from Kokoro.",
    "voice": "af_heart",
    "language": "en",
    "speed": 1.0,
    "format": "wav"
  }'
```

Stable Audio Open:

```bash
curl -X POST http://localhost:8002/generate/audio/stable-audio \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "warm ambient synth pad with soft rain",
    "negative_prompt": "distortion, clipping",
    "duration_seconds": 10,
    "steps": 100,
    "guidance_scale": 7.0,
    "format": "wav"
  }'
```

CosyVoice2:

```bash
curl -X POST http://localhost:8002/generate/tts/cosyvoice2 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a CosyVoice2 smoke test.",
    "seed": 42
  }'
```

Fish Speech:

```bash
curl -X POST http://localhost:8002/generate/tts/fish-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a Fish Speech smoke test.",
    "seed": 42
  }'
```

IndexTTS-2:

```bash
curl -X POST http://localhost:8002/generate/tts/indextts2 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is an IndexTTS two smoke test.",
    "seed": 42
  }'
```

After each audio request:

```bash
ls -lh /home/long/local-ai/outputs/audio | tail
```
