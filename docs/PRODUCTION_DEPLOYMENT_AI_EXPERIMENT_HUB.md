# Production Deployment Notes: AI Experiment Hub + GPU Inference

This repository deploys inference-plane service only.

## Required Split

1. Frontend -> control-plane API only
2. Control-plane job worker -> GPU inference server
3. Control-plane stores outputs/history and serves frontend polling

Do not expose GPU server base URL/API key to frontend.

## GPU Server Runtime Env

```bash
PUBLIC_BASE_URL=https://gpu.example.com
OUTPUT_ROOT=/srv/gpu-outputs
INFERENCE_TIMEOUT_SECONDS=300
HF_HOME=/srv/models/hf-cache
HF_HUB_CACHE=/srv/models/hf-cache/hub
```

## Control-Plane Runtime Env (backend only)

```bash
REMOTE_AUDIO_PROVIDER=gpu-inference-server
REMOTE_AUDIO_SERVER_BASE_URL=https://gpu.example.com
REMOTE_AUDIO_SERVER_API_KEY=<secret>
REMOTE_AUDIO_TIMEOUT_MS=300000
REMOTE_AUDIO_ALLOW_FALLBACK=true
REMOTE_AUDIO_DEFAULT_MODEL=kokoro-82m
```

Optional reuse from image provider:

```bash
REMOTE_IMAGE_SERVER_BASE_URL=https://gpu.example.com
REMOTE_IMAGE_SERVER_API_KEY=<secret>
```

## Health + Model Discovery

- `GET /health`
- `GET /models`

Use `/models` `available` field to disable non-runnable models upstream.

## Output Serving

- static outputs served under `/outputs/...`
- control-plane should ingest/republish outputs; frontend should not bind directly to raw GPU host

## Smoke Checklist

1. Call each configured audio endpoint with minimal payload.
2. Verify `output_url` under `/outputs/audio/...`.
3. Verify response includes `audio_kind`, `parameters_used`, `audio_duration_seconds`.
4. Verify unavailable models return `MODEL_NOT_LOADED` with `503`.
5. Verify unsupported fields return `UNSUPPORTED_PARAMETER` with `422`.

