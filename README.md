# Host-Inference-Models

Local GPU inference services for image generation and audio generation.

## Using the Gateway (Kokoro TTS and Distil-Whisper ASR)

Everything a user runs goes through the **model gateway** on port `9000`. Do not
call the per-model services (`audio-kokoro:8102`, `audio-legacy:8002`, ...)
directly — they are internal to the Docker network and have no auth.

Two models are covered here:

| Model id | Task | What it does |
|---|---|---|
| `kokoro-82m` | `text-to-speech` | Text in, a `.wav` file out (54 voices, 9 languages) |
| `distil-whisper-large-v3` | `automatic-speech-recognition` | Audio in, a transcript out |

### 1. Reach the gateway

The gateway only publishes `9000` on the GPU node's loopback. From your laptop,
open an SSH tunnel and leave it running:

```bash
ssh -N -L 9000:localhost:9000 <user>@gpu-farmi-004.rd.tuni.fi
```

Then `http://localhost:9000` is the gateway. Check it:

```bash
curl http://localhost:9000/health
```

`/health` is the only endpoint that needs no token.

### 2. Get a token

Every other endpoint needs a `Bearer` JWT signed with the gateway's shared
secret (`MODEL_GATEWAY_JWT_SECRET`). Ask a maintainer for the secret; it is not
in the repo.

The token carries two scopes:

- `permittedTasks` — which tasks you may **see** in `GET /models`
- `permittedModelIds` — which models you may **run** via `POST /generate`

Mint one with the helper script (stdlib only, no pip install):

```bash
export MODEL_GATEWAY_JWT_SECRET='<ask a maintainer>'

TOKEN=$(python3 scripts/mint_gateway_token.py \
  text-to-speech,automatic-speech-recognition \
  kokoro-82m,distil-whisper-large-v3)
```

Tokens expire after 15 minutes by default; pass a third argument for a
different TTL in seconds. Re-run the script when you get
`{"error": {"type": "Unauthorized"}}`.

List what your token can see:

```bash
curl -s http://localhost:9000/models \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

Each model entry carries a `fields` block — that is the authoritative list of
parameters and their bounds for that model.

### 3. Kokoro TTS — text to speech

```bash
curl -s -X POST http://localhost:9000/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro-82m",
    "input": { "text": "Hello from the GPT-Lab inference gateway." },
    "parameters": { "voice": "af_heart", "language": "en", "speed": 1.0 }
  }'
```

Response (trimmed):

```json
{
  "success": true,
  "status": "completed",
  "model_id": "kokoro-82m",
  "outputType": "audio",
  "outputUrl": "http://localhost:9000/outputs/audio/tts_<hex>.wav",
  "mime_type": "audio/wav",
  "audio_duration_seconds": 3.1,
  "sample_rate": 24000,
  "duration_ms": 812
}
```

Download the result (the `/outputs/` mount needs no token):

```bash
curl -sO http://localhost:9000/outputs/audio/tts_<hex>.wav
```

**Parameters** (`kokoro-82m`):

| Field | Default | Range / notes |
|---|---|---|
| `text` | — | required, max 12000 chars |
| `voice` | `af_heart` | see voice list below |
| `language` | `en` | `en`, `en-gb`, `es`, `fr`, `hi`, `it`, `ja`, `pt`, `zh` |
| `speed` | `1.0` | `0.5`–`2.0` |
| `format` | `wav` | `wav` only |
| `sample_rate` | `24000` | `24000` only |
| `lang_code` | derived from `language` | single letter, overrides `language` |

**Voices.** The prefix encodes language and gender — `a` American English,
`b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian,
`j` Japanese, `p` Portuguese, `z` Chinese; second letter `f` female, `m` male.
Pick a voice whose prefix matches `language`, or the phonemes will not line up.

```
af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova
af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam am_michael
am_onyx am_puck am_santa bf_alice bf_emma bf_isabella bf_lily bm_daniel
bm_fable bm_george bm_lewis ef_dora em_alex em_santa ff_siwis hf_alpha hf_beta
hm_omega hm_psi if_sara im_nicola jf_alpha jf_gongitsune jf_nezumi jf_tebukuro
jm_kumo pf_dora pm_alex pm_santa zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi
zm_yunjian zm_yunxia zm_yunxi zm_yunyang
```

There is also `kokoro-82m-onnx` (CPU ONNX build): same `text`/`voice`/`format`,
`speed` range `0.1`–`5.0`, no `language`/`sample_rate`/`lang_code`.

### 4. Distil-Whisper Large v3 — speech to text

The `audio` field takes **base64-encoded audio** (a bare base64 string or a
`data:` URI). Audio is resampled to 16 kHz internally, so any sample rate is
fine. Build the request with Python rather than shell interpolation — a base64
WAV blows past the shell's argument limit:

```bash
python3 - <<'PY'
import base64, json, os, urllib.request

audio = base64.b64encode(open("sample.wav", "rb").read()).decode()
body = json.dumps({
    "model": "distil-whisper-large-v3",
    "input": {"audio": audio},
    "parameters": {"language": "en"},
}).encode()

req = urllib.request.Request(
    "http://localhost:9000/generate",
    data=body,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['TOKEN']}",
    },
)
print(json.load(urllib.request.urlopen(req, timeout=600))["transcript"])
PY
```

Response (trimmed):

```json
{
  "success": true,
  "status": "completed",
  "model_id": "distil-whisper-large-v3",
  "outputType": "text",
  "transcript": "hello from the gpt lab inference gateway",
  "text": "hello from the gpt lab inference gateway",
  "language": "en",
  "task": "transcribe",
  "duration_ms": 1430
}
```

**Parameters**:

| Field | Default | Notes |
|---|---|---|
| `audio` | — | required; base64 string, `data:` URI, or a path the service can read |
| `language` | auto-detect | ISO code, e.g. `en`, `fi`, `sv` |
| `format` | `json` | `json` only |

The model always transcribes; it does not translate. For
translation-to-English use `whisper-large-v3`, which accepts `task`.

**Larger files.** Instead of inlining base64, upload the clip once and pass the
returned server-side path (`.wav` / `.flac` only, no token needed):

```bash
curl -s -X POST http://localhost:9000/uploads/audio -F file=@sample.wav
# {"path": "/outputs/uploads/ref_<hex>.wav", "url": "http://localhost:9000/outputs/uploads/ref_<hex>.wav"}
```

Then send `"input": {"audio": "/outputs/uploads/ref_<hex>.wav"}`. The gateway
and the ASR service share that `/outputs` mount, so the path resolves.

### 5. Request and response shape

Every `POST /generate` body is the same four keys:

```json
{
  "model": "<registry model id>",
  "input": { "...": "the primary payload — text, audio, prompt" },
  "parameters": { "...": "tuning knobs" },
  "request_id": "optional; echoed back for correlation"
}
```

`input` and `parameters` are merged before the call reaches the model service,
so the split is for readability only. A flat body (no `input`/`parameters`)
also works.

Failures return a non-2xx status and:

```json
{
  "success": false,
  "error": { "type": "ValidationError", "message": "..." },
  "model": "kokoro-82m",
  "metadata": { "request_id": "..." }
}
```

| Status | `error.type` | Usual cause |
|---|---|---|
| 400 | `ValidationError` | field out of range, unknown field, bad `input`/`parameters` |
| 401 | `Unauthorized` | missing, malformed, or expired token |
| 403 | `ScopeDenied` | model not in your `permittedModelIds`, task not in `permittedTasks` |
| 404 | `UnknownModel` | model id not in the registry (`details.available_models` lists yours) |
| 503 | `ServiceUnavailable` | the model container is down |
| 504 | `ServiceUnavailable` | generation exceeded `GATEWAY_TIMEOUT_SECONDS` (default 600s) |

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

## Docker Model Gateway

Docker images contain service code and runtime dependencies only. Model weights, Hugging Face cache, Torch cache, and generated outputs live in mounted host folders:

```bash
mkdir -p model-cache outputs
docker compose -f docker-compose.yml -f docker-compose.apple.yml build
docker compose -f docker-compose.yml -f docker-compose.apple.yml up -d
curl http://localhost:9000/health
curl http://localhost:9000/models -H "Authorization: Bearer $TOKEN"
```

Only `model-gateway` publishes port `9000`; model services stay on internal Docker service names.

The gateway owns model routing in `model-gateway/registry.yaml`. Add new models there by setting `task`, `modality`, `service`, `endpoint`, and supported `fields`; GPT-Lab backend does not need a routing change.

Gateway inference contract (every endpoint except `/health`, `/metrics` and
`/outputs/...` requires a scoped bearer token -- see
[Using the Gateway](#using-the-gateway-kokoro-tts-and-distil-whisper-asr)):

```bash
curl -X POST http://localhost:9000/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flux-1-schnell",
    "input": { "prompt": "a cinematic mountain lake" },
    "parameters": { "width": 1024, "height": 1024, "steps": 4 },
    "request_id": "optional-backend-job-id"
  }'
```

Gateway endpoints:

- `GET /health` -- no token
- `GET /models` -- scoped by `permittedTasks`
- `POST /generate` -- scoped by `permittedModelIds`
- `GET /jobs/{job_id}` -- scoped to your tenant
- `POST /uploads/audio` -- store a `.wav`/`.flac` clip, no token
- `GET /outputs/...` -- generated files, no token

Storage inspection:

```bash
scripts/docker_storage_report.sh
docker system df
docker builder prune
docker system prune
```

Aggressive cleanup removes unused images and may require rebuilding:

```bash
docker system prune -a
```

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

Maintainer checks that run on the GPU node and hit the model services directly,
bypassing the gateway. Users should go through the gateway instead.

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
