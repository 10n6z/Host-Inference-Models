# GPU Inference Server API Contract

Inference plane only.

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

Returns model metadata + field constraints.

- `available: false` means endpoint exists but should be disabled by control plane UI.
- `audioKind` distinguishes TTS (`tts`) from text-to-audio (`text-to-audio`).

Example snippet:

```json
{
  "models": [
    {
      "id": "kokoro-82m",
      "displayName": "Kokoro-82M",
      "modality": "audio",
      "audioKind": "tts",
      "endpoint": "/generate/tts/kokoro",
      "available": true,
      "fields": {
        "text": { "type": "string", "required": true, "max_length": 12000 },
        "voice": { "type": "string", "default": "af_heart" }
      }
    },
    {
      "id": "stable-audio-open-1.0",
      "displayName": "Stable Audio Open 1.0",
      "modality": "audio",
      "audioKind": "text-to-audio",
      "endpoint": "/generate/audio/stable-audio-open",
      "available": true
    }
  ]
}
```

## Image Endpoints

### POST /generate/image/flux

- `prompt` required (max 4000)
- `width`/`height`: 512-2048, multiple of 8
- `steps`: 1-20
- `guidance_scale`: must be `0`
- `max_sequence_length`: max 256
- `seed` + `random_seed`
- `num_images`: max `1`

### POST /generate/image/sd35

- `prompt` required
- `negative_prompt` optional
- `width`/`height`: 512-2048, multiple of 8
- `steps`: 1-60
- `guidance_scale`: 0-20
- `seed` + `random_seed`
- `num_images`: max `1`

### POST /generate/image/auraflow

- `prompt` required
- `negative_prompt` optional
- `width`/`height`: 512-2048, multiple of 8
- `steps`: 1-60
- `guidance_scale`: 0-20
- `seed` + `random_seed`
- `num_images`: max `1`

### POST /generate/image/openflux

- `prompt` required
- `width`/`height`: 512-2048, multiple of 8
- `steps`: 1-60
- `guidance_scale`: 0-20
- `max_sequence_length`: max 512
- `seed` + `random_seed`
- `num_images`: max `1`

## Audio Endpoints

### POST /generate/tts/kokoro

- `text` required, max 12000
- `voice` default `af_heart`
- `language` default `en`
- `speed`: 0.5-2.0
- `format`: `wav`
- `sample_rate`: `24000`
- `lang_code` optional override

Language mapping to Kokoro `lang_code`:

- `en`/`en-us` -> `a`
- `en-gb` -> `b`
- `es` -> `e`
- `fr-fr` -> `f`
- `hi` -> `h`
- `it` -> `i`
- `ja` -> `j`
- `pt-br` -> `p`
- `zh`/`zh-cn` -> `z`

### POST /generate/tts/fish-speech

- `text` required, max 12000
- `language` enum: `en, zh, ja, de, fr, es, ko, ar, ru, nl, it, pl, pt`
- `voice` default `default`
- `reference_audio_id` optional
- `speed`: 0.5-2.0
- `format`: `wav`
- `sample_rate` optional (8000-96000)

### POST /generate/tts/cosyvoice2

- `text` required, max 12000
- `language` enum: `zh, en, ja, ko, de, es, fr, it, ru`
- `speaker` default `default`
- `reference_audio_id` optional
- `instruction` optional, max 500
- `speed`: 0.5-2.0
- `stream` boolean (accepted; response still file output)
- `format`: `wav`

### POST /generate/tts/indextts2

- `text` required, max 12000
- `speaker` default `default`
- `reference_audio_id` optional
- `emotion` optional
- `duration_control` optional 0.5-120.0
- `speed`: 0.5-2.0
- `format`: `wav`

### POST /generate/audio/stable-audio-open

Text-to-audio (music/sound), not standard TTS.

- `prompt` required, max 4000
- `negative_prompt` optional
- `duration_seconds`: 1-47
- `steps`: 1-250
- `guidance_scale`: 0-25
- `seed` + `random_seed`
- `format`: `wav`

## Success Response Shape

Image + audio responses return:

```json
{
  "success": true,
  "model_id": "stable-audio-open-1.0",
  "modality": "audio",
  "audio_kind": "text-to-audio",
  "output_url": "/outputs/audio/aud_xxx.wav",
  "public_output_url": "http://host:8001/outputs/audio/aud_xxx.wav",
  "file_name": "aud_xxx.wav",
  "mime_type": "audio/wav",
  "parameters_used": {},
  "duration_ms": 12345,
  "audio_duration_seconds": 4.2,
  "duration_seconds": 4.2,
  "sample_rate": 44100,
  "created_at": "2026-05-19T00:00:00Z"
}
```

`duration_seconds` kept for backward compatibility; prefer `audio_duration_seconds`.

## Curl Smoke Examples

Kokoro:

```bash
curl -X POST "$GPU_SERVER_URL/generate/tts/kokoro" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from AI Experiment Hub.",
    "voice": "af_heart",
    "language": "en",
    "speed": 1.0,
    "format": "wav"
  }'
```

Fish Speech:

```bash
curl -X POST "$GPU_SERVER_URL/generate/tts/fish-speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from Fish Speech.",
    "language": "en",
    "voice": "default",
    "speed": 1.0,
    "format": "wav"
  }'
```

CosyVoice2:

```bash
curl -X POST "$GPU_SERVER_URL/generate/tts/cosyvoice2" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from CosyVoice two.",
    "language": "en",
    "speaker": "default",
    "speed": 1.0,
    "format": "wav",
    "stream": false
  }'
```

IndexTTS-2:

```bash
curl -X POST "$GPU_SERVER_URL/generate/tts/indextts2" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from Index TTS two.",
    "speaker": "default",
    "emotion": "neutral",
    "speed": 1.0,
    "format": "wav"
  }'
```

Stable Audio Open:

```bash
curl -X POST "$GPU_SERVER_URL/generate/audio/stable-audio-open" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A short cinematic ambient drone with soft bells",
    "duration_seconds": 10,
    "steps": 50,
    "guidance_scale": 7.0,
    "seed": 42,
    "random_seed": false,
    "format": "wav"
  }'
```

