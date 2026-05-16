# GPU Inference Server API Contract

This server is inference-plane only.

- No auth
- No queue/retry/job history/database
- No arbitrary model IDs or file paths
- Backend control plane validates first; GPU server validates defensively

## Base

- Health: `GET /health`
- Model metadata: `GET /models`
- Static outputs: `GET /outputs/...`

## Error Shape

All request/validation/runtime failures return:

```json
{
  "success": false,
  "code": "VALIDATION_ERROR",
  "message": "width: Input should be less than or equal to 2048",
  "details": {}
}
```

Error codes:

- `VALIDATION_ERROR`
- `UNSUPPORTED_PARAMETER`
- `MODEL_NOT_LOADED`
- `GENERATION_FAILED`
- `CUDA_OUT_OF_MEMORY`
- `OUTPUT_SAVE_FAILED`
- `TIMEOUT`

## GET /models

Returns dynamic model metadata and supported request fields.

```json
{
  "models": [
    {
      "id": "flux-1-schnell",
      "displayName": "FLUX.1 Schnell",
      "modality": "image",
      "endpoint": "/generate/image/flux",
      "fields": {
        "prompt": { "type": "string", "required": true },
        "width": { "type": "integer", "default": 1024, "min": 512, "max": 2048, "step": 8 },
        "height": { "type": "integer", "default": 1024, "min": 512, "max": 2048, "step": 8 },
        "steps": { "type": "integer", "default": 4, "min": 1, "max": 20 },
        "guidance_scale": { "type": "number", "default": 0.0, "min": 0.0, "max": 0.0 },
        "seed": { "type": "integer", "required": false },
        "random_seed": { "type": "boolean", "default": true },
        "max_sequence_length": { "type": "integer", "default": 256, "max": 256 }
      }
    }
  ],
  "field_catalog": {
    "future_audio_video_fields": [
      "duration",
      "guidance_scale",
      "seed",
      "reference_audio",
      "reference_image",
      "sample_rate",
      "fps",
      "resolution",
      "num_frames"
    ]
  }
}
```

## Common Image Request Rules

- `prompt` required
- `prompt` max length: 4000
- `width`/`height`: 512-2048, multiple of 8
- `steps`: model-specific min/max
- `guidance_scale`: model-specific min/max
- `num_images`: currently max `1` (MVP; no batching)
- `random_seed=true`: server generates seed and returns it in `parameters_used.seed`
- `random_seed=false`: `seed` required

`steps` maps to pipeline `num_inference_steps`.

## Image Endpoints

### POST /generate/image/flux

Fields:

- `prompt`
- `width`, `height`
- `steps` (1-20, default 4)
- `guidance_scale` (must be 0)
- `seed`, `random_seed`
- `max_sequence_length` (max 256)
- `num_images` (max 1)

### POST /generate/image/sd35

Fields:

- `prompt`
- `negative_prompt`
- `width`, `height`
- `steps` (1-60, default 28)
- `guidance_scale` (0-20, default 7.0)
- `seed`, `random_seed`
- `num_images` (max 1)

### POST /generate/image/auraflow

Fields:

- `prompt`
- `negative_prompt`
- `width`, `height`
- `steps` (1-60, default 30)
- `guidance_scale` (0-20, default 5.0)
- `seed`, `random_seed`
- `num_images` (max 1)

### POST /generate/image/openflux

Fields:

- `prompt`
- `width`, `height`
- `steps` (1-60, default 28)
- `guidance_scale` (0-20, default 7.0)
- `max_sequence_length` (max 512)
- `seed`, `random_seed`
- `num_images` (max 1)

Note: current OpenFLUX endpoint uses FluxPipeline-compatible args from local implementation. Unsupported request keys are rejected.

## Audio Endpoint

### POST /generate/tts/kokoro

Fields:

- `text` required, max length 12000
- `voice` optional (default `af_heart`)
- `language` optional (default `en`)
- `speed` optional (0.5-2.0, default 1.0)
- `format` optional (`wav` only)
- `sample_rate` optional (`24000` only)
- `lang_code` optional (backward compatibility override)

Language maps to Kokoro `lang_code` (examples):

- `en`/`en-us` -> `a`
- `en-gb` -> `b`
- `es` -> `e`
- `fr-fr` -> `f`
- `hi` -> `h`
- `it` -> `i`
- `ja` -> `j`
- `pt-br` -> `p`
- `zh` -> `z`

## Success Response Shape

Image and audio endpoints return:

```json
{
  "success": true,
  "model_id": "flux-1-schnell",
  "modality": "image",
  "output_url": "/outputs/images/img_xxx.png",
  "public_output_url": "http://host:8001/outputs/images/img_xxx.png",
  "file_name": "img_xxx.png",
  "mime_type": "image/png",
  "parameters_used": {
    "prompt": "...",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance_scale": 0,
    "seed": 123
  },
  "duration_ms": 12345,
  "created_at": "2026-05-16T00:00:00Z"
}
```

Audio responses additionally include:

- `duration_seconds` (if available)
- `parameters_used.voice`
- `parameters_used.language`
- `parameters_used.speed`
- `parameters_used.sample_rate`

## Backward Compatibility

Prompt-only requests still valid.

Example:

```json
{ "prompt": "A cat holding a sign that says hello world" }
```

Server applies model defaults.

## Curl Smoke Examples

FLUX:

```bash
curl -X POST "$GPU_SERVER_URL/generate/image/flux" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat holding a sign that says hello world",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance_scale": 0,
    "seed": 42,
    "random_seed": false
  }'
```

SD3.5:

```bash
curl -X POST "$GPU_SERVER_URL/generate/image/sd35" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cinematic robot in a neon city",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "guidance_scale": 5.0,
    "seed": 123,
    "random_seed": false
  }'
```

AuraFlow:

```bash
curl -X POST "$GPU_SERVER_URL/generate/image/auraflow" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cozy paper craft diorama workspace",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 5.0,
    "seed": 7,
    "random_seed": false
  }'
```

Kokoro TTS:

```bash
curl -X POST "$GPU_SERVER_URL/generate/tts/kokoro" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from the AI Experiment Hub.",
    "voice": "af_heart",
    "language": "en",
    "speed": 1.0,
    "format": "wav"
  }'
```

## Integration Notes

- Frontend must not call GPU server directly.
- Backend control plane should call `GET /models` for dynamic forms + allowlist validation.
- GPU server remains defensive: rejects unknown fields, out-of-range values, and unsupported settings.
