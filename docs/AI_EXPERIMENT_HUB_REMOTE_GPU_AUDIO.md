# AI Experiment Hub Remote GPU Audio

Scope of this repo: GPU inference plane only.

Control-plane responsibilities stay in backend API/job worker:

- request validation and model fallback
- queue/job status/history/retry
- storage persistence
- provider catalog for frontend
- GPU routing and credentials

Frontend must never call GPU server directly.

## Backend-Only Env Vars (control-plane)

```bash
REMOTE_AUDIO_PROVIDER=gpu-inference-server
REMOTE_AUDIO_SERVER_BASE_URL=<GPU inference server base URL>
REMOTE_AUDIO_SERVER_API_KEY=<optional backend-only token>
REMOTE_AUDIO_TIMEOUT_MS=300000
REMOTE_AUDIO_ALLOW_FALLBACK=true
REMOTE_AUDIO_DEFAULT_MODEL=kokoro-82m
```

Fallback reuse if same GPU host as image:

- `REMOTE_IMAGE_SERVER_BASE_URL`
- `REMOTE_IMAGE_SERVER_API_KEY`

Audio should prefer `REMOTE_AUDIO_*`.

## GPU Endpoint Map (fixed, no arbitrary routing)

- `kokoro-82m` -> `/generate/tts/kokoro`
- `fish-speech-v1.5` -> `/generate/tts/fish-speech`
- `cosyvoice2-0.5b` -> `/generate/tts/cosyvoice2`
- `indextts-2` -> `/generate/tts/indextts2`
- `stable-audio-open-1.0` -> `/generate/audio/stable-audio-open`

## Model Kinds

- TTS: Kokoro, Fish Speech, CosyVoice2, IndexTTS-2
- Text-to-audio: Stable Audio Open 1.0 (music/sound), not standard TTS

## Request Schema Notes

TTS common:

- required `text`
- optional voice/speaker/language
- optional `speed` where supported
- `format` currently `wav` only in this GPU server contract
- optional `reference_audio_id` on models that support voice conditioning

Stable Audio Open:

- required `prompt`
- `duration_seconds` max 47 (current contract)
- `steps`, `guidance_scale`
- `seed` + `random_seed`

## Response Metadata

Audio endpoints return:

- `model_id`, `modality`, `audio_kind`
- `output_url`, `public_output_url`, `file_name`, `mime_type`
- `parameters_used`
- `duration_ms`
- `audio_duration_seconds` (plus compatibility `duration_seconds`)
- `sample_rate`

## Security Rules

- keep GPU URL and API key backend-only
- do not expose arbitrary GPU endpoints/model IDs/file paths
- reject unsupported fields by schema (`extra="forbid"`)
- do not persist secrets in DB/history/logs

