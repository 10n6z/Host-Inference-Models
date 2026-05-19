import mimetypes
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from runners.text_to_image.auraflow import AuraFlowRunner
from runners.text_to_image.flux_schnell import FluxSchnellRunner
from runners.text_to_image.openflux import OpenFluxRunner
from runners.text_to_image.sd35_medium import SD35MediumRunner
from runners.text_to_audio.stable_audio_open_runner import StableAudioOpenRunner
from runners.text_to_speech.cosyvoice2_runner import CosyVoice2Runner
from runners.text_to_speech.fish_speech_runner import FishSpeechRunner
from runners.text_to_speech.indextts2_runner import IndexTTS2Runner
from runners.text_to_speech.kokoro_runner import KokoroRunner

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

HF_HOME = Path(os.getenv("HF_HOME", BASE_DIR.parent / "models" / "hf-cache")).resolve()
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE", HF_HOME / "hub")).resolve()

OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", BASE_DIR.parent / "outputs")).resolve()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8001").rstrip("/")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "300"))

IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "images"
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audio"

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

PROMPT_MAX_LENGTH = 4000
TTS_TEXT_MAX_LENGTH = 12000
TTA_TEXT_MAX_LENGTH = 4000
IMAGE_MIN_SIZE = 512
IMAGE_MAX_SIZE = 2048
IMAGE_SIZE_STEP = 8
MAX_SEED = 2_147_483_647
MAX_NUM_IMAGES = 1
MAX_INSTRUCTION_LENGTH = 500
MAX_REF_AUDIO_ID_LENGTH = 200
MAX_SPEAKER_NAME_LENGTH = 100
STABLE_AUDIO_MIN_DURATION_SECONDS = 1
STABLE_AUDIO_MAX_DURATION_SECONDS = 47
STABLE_AUDIO_DEFAULT_SAMPLE_RATE = 44100
WAV_ONLY_PATTERN = "^(wav)$"

FISH_SPEECH_LANGUAGES = sorted({"en", "zh", "ja", "de", "fr", "es", "ko", "ar", "ru", "nl", "it", "pl", "pt"})
COSYVOICE2_LANGUAGES = sorted({"zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"})

KOKORO_LANGUAGE_TO_CODE = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "pt-br": "p",
    "zh": "z",
    "zh-cn": "z",
}

app = FastAPI(title="Local AI GPU Inference Server")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

flux_runner = FluxSchnellRunner()
sd35_runner = SD35MediumRunner()
auraflow_runner = AuraFlowRunner()
openflux_runner = OpenFluxRunner()
kokoro_runner = KokoroRunner()
fish_speech_runner = FishSpeechRunner()
cosyvoice2_runner = CosyVoice2Runner()
indextts2_runner = IndexTTS2Runner()
stable_audio_open_runner = StableAudioOpenRunner()


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: Optional[dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageSeedMixin(StrictRequestModel):
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    random_seed: bool = True

    @model_validator(mode="after")
    def validate_seed_requirements(self):
        if not self.random_seed and self.seed is None:
            raise ValueError("seed is required when random_seed is false.")
        return self


class FluxRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(4, ge=1, le=20)
    guidance_scale: float = Field(0.0, ge=0.0, le=0.0)
    max_sequence_length: int = Field(256, ge=1, le=256)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class SD35Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(28, ge=1, le=60)
    guidance_scale: float = Field(7.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class AuraFlowRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class OpenFluxRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(28, ge=1, le=60)
    guidance_scale: float = Field(7.0, ge=0.0, le=20.0)
    max_sequence_length: int = Field(512, ge=1, le=512)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class KokoroRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    voice: str = Field("af_heart", min_length=1, max_length=100)
    language: str = Field("en", min_length=2, max_length=10)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    sample_rate: int = Field(24000, ge=24000, le=24000)
    lang_code: Optional[str] = Field(default=None, min_length=1, max_length=1)


class FishSpeechRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    language: str = Field("en", min_length=2, max_length=10)
    voice: str = Field("default", min_length=1, max_length=MAX_SPEAKER_NAME_LENGTH)
    reference_audio_id: Optional[str] = Field(default=None, min_length=1, max_length=MAX_REF_AUDIO_ID_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    sample_rate: Optional[int] = Field(default=None, ge=8000, le=96000)

    @model_validator(mode="after")
    def validate_language(self):
        language = self.language.lower()
        if language not in FISH_SPEECH_LANGUAGES:
            raise ValueError(f"language must be one of: {', '.join(FISH_SPEECH_LANGUAGES)}")
        return self


class CosyVoice2Request(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    language: str = Field("en", min_length=2, max_length=10)
    speaker: str = Field("default", min_length=1, max_length=MAX_SPEAKER_NAME_LENGTH)
    reference_audio_id: Optional[str] = Field(default=None, min_length=1, max_length=MAX_REF_AUDIO_ID_LENGTH)
    instruction: Optional[str] = Field(default=None, max_length=MAX_INSTRUCTION_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    stream: bool = False

    @model_validator(mode="after")
    def validate_language(self):
        language = self.language.lower()
        if language not in COSYVOICE2_LANGUAGES:
            raise ValueError(f"language must be one of: {', '.join(COSYVOICE2_LANGUAGES)}")
        return self


class IndexTTS2Request(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    speaker: str = Field("default", min_length=1, max_length=MAX_SPEAKER_NAME_LENGTH)
    reference_audio_id: Optional[str] = Field(default=None, min_length=1, max_length=MAX_REF_AUDIO_ID_LENGTH)
    emotion: Optional[str] = Field(default=None, min_length=1, max_length=100)
    duration_control: Optional[float] = Field(default=None, ge=0.5, le=120.0)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


class StableAudioOpenRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=TTA_TEXT_MAX_LENGTH)
    duration_seconds: float = Field(10.0, ge=STABLE_AUDIO_MIN_DURATION_SECONDS, le=STABLE_AUDIO_MAX_DURATION_SECONDS)
    steps: int = Field(50, ge=1, le=250)
    guidance_scale: float = Field(7.0, ge=0.0, le=25.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    negative_prompt: Optional[str] = Field(default=None, max_length=TTA_TEXT_MAX_LENGTH)


def _field_spec(
    field_type: str,
    *,
    required: Optional[bool] = None,
    default: Any = None,
    minimum: Any = None,
    maximum: Any = None,
    step: Any = None,
    enum: Optional[list[Any]] = None,
    max_length: Optional[int] = None,
    description: Optional[str] = None,
):
    data: dict[str, Any] = {"type": field_type}
    if required is not None:
        data["required"] = required
    if default is not None:
        data["default"] = default
    if minimum is not None:
        data["min"] = minimum
    if maximum is not None:
        data["max"] = maximum
    if step is not None:
        data["step"] = step
    if enum is not None:
        data["enum"] = enum
    if max_length is not None:
        data["max_length"] = max_length
    if description:
        data["description"] = description
    return data


def _audio_response(
    *,
    model_id: str,
    audio_kind: str,
    output_id: str,
    output_path: Path,
    parameters_used: dict[str, Any],
    duration_ms: int,
    audio_meta: Optional[dict[str, Any]] = None,
):
    relative_url = _output_url("audio", output_id)
    sample_rate = None
    audio_duration_seconds = None

    if isinstance(audio_meta, dict):
        raw_sample_rate = audio_meta.get("sample_rate")
        if raw_sample_rate is not None:
            sample_rate = int(raw_sample_rate)

        raw_duration = audio_meta.get("audio_duration_seconds", audio_meta.get("duration_seconds"))
        if raw_duration is not None:
            audio_duration_seconds = float(raw_duration)

    return {
        "success": True,
        "model_id": model_id,
        "modality": "audio",
        "audio_kind": audio_kind,
        "output_url": relative_url,
        "public_output_url": _public_output_url(relative_url),
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "audio/wav",
        "parameters_used": parameters_used,
        "duration_ms": duration_ms,
        "audio_duration_seconds": audio_duration_seconds,
        "duration_seconds": audio_duration_seconds,
        "sample_rate": sample_rate,
        "created_at": _utc_now_iso(),
    }


def _model_registry() -> list[dict[str, Any]]:
    return [
        {
            "id": "flux-1-schnell",
            "displayName": "FLUX.1 Schnell",
            "modality": "image",
            "endpoint": "/generate/image/flux",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=4, minimum=1, maximum=20),
                "guidance_scale": _field_spec("number", default=0.0, minimum=0.0, maximum=0.0),
                "max_sequence_length": _field_spec("integer", default=256, minimum=1, maximum=256),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
            "notes": [
                "FLUX.1-schnell is timestep-distilled: guidance_scale must be 0.",
                "max_sequence_length cannot exceed 256.",
            ],
        },
        {
            "id": "stable-diffusion-3.5-medium",
            "displayName": "Stable Diffusion 3.5 Medium",
            "modality": "image",
            "endpoint": "/generate/image/sd35",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=28, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.0, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
        },
        {
            "id": "auraflow-v0.3",
            "displayName": "AuraFlow v0.3",
            "modality": "image",
            "endpoint": "/generate/image/auraflow",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=5.0, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
        },
        {
            "id": "openflux-1",
            "displayName": "OpenFLUX.1",
            "modality": "image",
            "endpoint": "/generate/image/openflux",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=28, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.0, minimum=0.0, maximum=20.0),
                "max_sequence_length": _field_spec("integer", default=512, minimum=1, maximum=512),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
            "notes": [
                "This endpoint uses FluxPipeline-compatible arguments in current server implementation.",
                "Unsupported request keys are rejected via schema validation.",
            ],
        },
        {
            "id": "kokoro-82m",
            "displayName": "Kokoro-82M",
            "modality": "audio",
            "audioKind": "tts",
            "endpoint": "/generate/tts/kokoro",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "voice": _field_spec("string", default="af_heart"),
                "language": _field_spec(
                    "string",
                    default="en",
                    enum=sorted({"en", "en-us", "en-gb", "es", "fr-fr", "hi", "it", "ja", "pt-br", "zh"}),
                ),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=2.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
                "sample_rate": _field_spec("integer", default=24000, minimum=24000, maximum=24000),
            },
            "capabilities": {
                "referenceAudio": False,
                "instruction": False,
                "emotion": False,
                "durationControl": False,
                "seed": False,
                "sampleRate": True,
            },
            "notes": [
                "Current runner writes WAV at 24000 Hz only.",
                "lang_code remains accepted for backward compatibility.",
            ],
        },
        {
            "id": "fish-speech-v1.5",
            "displayName": "Fish Speech v1.5",
            "modality": "audio",
            "audioKind": "tts",
            "endpoint": "/generate/tts/fish-speech",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "language": _field_spec("string", default="en", enum=FISH_SPEECH_LANGUAGES),
                "voice": _field_spec("string", default="default"),
                "reference_audio_id": _field_spec("string", required=False, max_length=MAX_REF_AUDIO_ID_LENGTH),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=2.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
                "sample_rate": _field_spec("integer", required=False, minimum=8000, maximum=96000),
            },
            "capabilities": {
                "referenceAudio": True,
                "instruction": False,
                "emotion": False,
                "durationControl": False,
                "seed": False,
                "sampleRate": True,
            },
            "notes": [
                "Configured as multilingual TTS contract.",
            ],
        },
        {
            "id": "cosyvoice2-0.5b",
            "displayName": "CosyVoice2-0.5B",
            "modality": "audio",
            "audioKind": "tts",
            "endpoint": "/generate/tts/cosyvoice2",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "language": _field_spec("string", default="en", enum=COSYVOICE2_LANGUAGES),
                "speaker": _field_spec("string", default="default"),
                "reference_audio_id": _field_spec("string", required=False, max_length=MAX_REF_AUDIO_ID_LENGTH),
                "instruction": _field_spec("string", required=False, max_length=MAX_INSTRUCTION_LENGTH),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=2.0),
                "stream": _field_spec("boolean", default=False),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
            "capabilities": {
                "referenceAudio": True,
                "instruction": True,
                "emotion": False,
                "durationControl": False,
                "seed": False,
                "sampleRate": False,
            },
            "notes": [
                "Configured as multilingual TTS contract with optional instruction.",
                "Streaming flag accepted; endpoint currently returns non-streaming file output only.",
            ],
        },
        {
            "id": "indextts-2",
            "displayName": "IndexTTS-2",
            "modality": "audio",
            "audioKind": "tts",
            "endpoint": "/generate/tts/indextts2",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "speaker": _field_spec("string", default="default"),
                "reference_audio_id": _field_spec("string", required=False, max_length=MAX_REF_AUDIO_ID_LENGTH),
                "emotion": _field_spec("string", required=False, max_length=100),
                "duration_control": _field_spec("number", required=False, minimum=0.5, maximum=120.0),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=2.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
            "capabilities": {
                "referenceAudio": True,
                "instruction": False,
                "emotion": True,
                "durationControl": True,
                "seed": False,
                "sampleRate": False,
            },
            "notes": [
                "Configured for zero-shot style inputs (speaker/reference).",
            ],
        },
        {
            "id": "stable-audio-open-1.0",
            "displayName": "Stable Audio Open 1.0",
            "modality": "audio",
            "audioKind": "text-to-audio",
            "endpoint": "/generate/audio/stable-audio-open",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=TTA_TEXT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=TTA_TEXT_MAX_LENGTH),
                "duration_seconds": _field_spec(
                    "number",
                    default=10.0,
                    minimum=STABLE_AUDIO_MIN_DURATION_SECONDS,
                    maximum=STABLE_AUDIO_MAX_DURATION_SECONDS,
                ),
                "steps": _field_spec("integer", default=50, minimum=1, maximum=250),
                "guidance_scale": _field_spec("number", default=7.0, minimum=0.0, maximum=25.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
            "capabilities": {
                "referenceAudio": False,
                "instruction": False,
                "emotion": False,
                "durationControl": True,
                "seed": True,
                "sampleRate": True,
            },
            "notes": [
                "Text-to-audio generation (music/sound), not standard TTS.",
                f"Duration capped at {STABLE_AUDIO_MAX_DURATION_SECONDS} seconds in this contract.",
                "Current response returns file output only (no streaming chunks).",
            ],
        },
    ]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _output_url(subdir: str, file_name: str) -> str:
    return f"/outputs/{subdir}/{file_name}"


def _public_output_url(relative_output_url: str) -> str:
    return f"{PUBLIC_BASE_URL}{relative_output_url}"


def _resolve_seed(random_seed: bool, seed: Optional[int]) -> int:
    if random_seed:
        return secrets.randbelow(MAX_SEED + 1)
    if seed is None:
        raise APIError(
            code="VALIDATION_ERROR",
            message="seed is required when random_seed is false.",
            status_code=422,
            details={"seed": "Provide seed or set random_seed=true."},
        )
    return int(seed)


def _run_with_timeout(func, *args, timeout_seconds: float = INFERENCE_TIMEOUT_SECONDS, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise APIError(
                code="TIMEOUT",
                message=f"Generation timed out after {int(timeout_seconds)} seconds.",
                status_code=504,
            ) from exc


def _check_output_exists(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise APIError(
            code="OUTPUT_SAVE_FAILED",
            message="Model run completed but output file was not saved.",
            status_code=500,
            details={"output_path": str(path)},
        )


def _map_runtime_error(exc: Exception) -> APIError:
    message = str(exc)
    lower = message.lower()

    if isinstance(exc, APIError):
        return exc

    if isinstance(exc, FileNotFoundError) or "model folder not found" in lower:
        return APIError("MODEL_NOT_LOADED", message, 503)

    if "cuda out of memory" in lower or "out of memory" in lower:
        return APIError("CUDA_OUT_OF_MEMORY", message, 507)

    return APIError("GENERATION_FAILED", message, 500)


def _validation_message(errors: list[dict[str, Any]]) -> str:
    first = errors[0] if errors else {}
    loc = first.get("loc", [])
    loc_text = ".".join(str(part) for part in loc if part not in ("body",))
    msg = first.get("msg", "Invalid request body.")
    if loc_text:
        return f"{loc_text}: {msg}"
    return msg


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError):
    errors = exc.errors()
    is_unsupported = any(err.get("type") == "extra_forbidden" for err in errors)
    code = "UNSUPPORTED_PARAMETER" if is_unsupported else "VALIDATION_ERROR"

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "code": code,
            "message": _validation_message(errors),
            "details": {"errors": errors},
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "local-ai-gpu-inference-server",
        "created_at": _utc_now_iso(),
    }


@app.get("/models")
def models():
    return {
        "models": _model_registry(),
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
                "num_frames",
            ]
        },
    }


@app.post("/generate/image/flux")
def generate_flux(req: FluxRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            flux_runner.generate,
            prompt=req.prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            max_sequence_length=req.max_sequence_length,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    relative_url = _output_url("images", output_id)

    return {
        "success": True,
        "model_id": "flux-1-schnell",
        "modality": "image",
        "output_url": relative_url,
        "public_output_url": _public_output_url(relative_url),
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "image/png",
        "parameters_used": {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "max_sequence_length": req.max_sequence_length,
            "num_images": req.num_images,
        },
        "duration_ms": duration_ms,
        "created_at": _utc_now_iso(),
    }


@app.post("/generate/image/sd35")
def generate_sd35(req: SD35Request):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            sd35_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    relative_url = _output_url("images", output_id)

    return {
        "success": True,
        "model_id": "stable-diffusion-3.5-medium",
        "modality": "image",
        "output_url": relative_url,
        "public_output_url": _public_output_url(relative_url),
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "image/png",
        "parameters_used": {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "num_images": req.num_images,
        },
        "duration_ms": duration_ms,
        "created_at": _utc_now_iso(),
    }


@app.post("/generate/image/auraflow")
def generate_auraflow(req: AuraFlowRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            auraflow_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    relative_url = _output_url("images", output_id)

    return {
        "success": True,
        "model_id": "auraflow-v0.3",
        "modality": "image",
        "output_url": relative_url,
        "public_output_url": _public_output_url(relative_url),
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "image/png",
        "parameters_used": {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "num_images": req.num_images,
        },
        "duration_ms": duration_ms,
        "created_at": _utc_now_iso(),
    }


@app.post("/generate/image/openflux")
def generate_openflux(req: OpenFluxRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            openflux_runner.generate,
            prompt=req.prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            max_sequence_length=req.max_sequence_length,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    relative_url = _output_url("images", output_id)

    return {
        "success": True,
        "model_id": "openflux-1",
        "modality": "image",
        "output_url": relative_url,
        "public_output_url": _public_output_url(relative_url),
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "image/png",
        "parameters_used": {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "max_sequence_length": req.max_sequence_length,
            "num_images": req.num_images,
        },
        "duration_ms": duration_ms,
        "created_at": _utc_now_iso(),
    }


@app.post("/generate/tts/kokoro")
def generate_kokoro(req: KokoroRequest):
    format_normalized = req.format.lower()
    output_id = f"aud_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    language = req.language.lower()
    lang_code = req.lang_code or KOKORO_LANGUAGE_TO_CODE.get(language)

    if lang_code is None:
        raise APIError(
            code="VALIDATION_ERROR",
            message=f"Unsupported language '{req.language}'.",
            status_code=422,
            details={"supported_languages": sorted(KOKORO_LANGUAGE_TO_CODE.keys())},
        )

    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            kokoro_runner.generate,
            text=req.text,
            output_path=str(output_path),
            voice=req.voice,
            speed=req.speed,
            lang_code=lang_code,
            sample_rate=req.sample_rate,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)

    parameters_used = {
        "text": req.text,
        "voice": req.voice,
        "language": language,
        "lang_code": lang_code,
        "speed": req.speed,
        "format": format_normalized,
        "sample_rate": req.sample_rate,
    }

    if isinstance(audio_meta, dict) and "sample_rate" in audio_meta:
        parameters_used["sample_rate"] = int(audio_meta["sample_rate"])

    return _audio_response(
        model_id="kokoro-82m",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used=parameters_used,
        duration_ms=duration_ms,
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/fish-speech")
def generate_fish_speech(req: FishSpeechRequest):
    format_normalized = req.format.lower()
    language = req.language.lower()
    output_id = f"aud_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            fish_speech_runner.generate,
            text=req.text,
            output_path=str(output_path),
            language=language,
            voice=req.voice,
            reference_audio_id=req.reference_audio_id,
            speed=req.speed,
            sample_rate=req.sample_rate,
            format=format_normalized,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    return _audio_response(
        model_id="fish-speech-v1.5",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "language": language,
            "voice": req.voice,
            "reference_audio_id": req.reference_audio_id,
            "speed": req.speed,
            "format": format_normalized,
            "sample_rate": req.sample_rate,
        },
        duration_ms=duration_ms,
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/cosyvoice2")
def generate_cosyvoice2(req: CosyVoice2Request):
    format_normalized = req.format.lower()
    language = req.language.lower()
    output_id = f"aud_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            cosyvoice2_runner.generate,
            text=req.text,
            output_path=str(output_path),
            language=language,
            speaker=req.speaker,
            reference_audio_id=req.reference_audio_id,
            instruction=req.instruction,
            speed=req.speed,
            format=format_normalized,
            stream=req.stream,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    return _audio_response(
        model_id="cosyvoice2-0.5b",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "language": language,
            "speaker": req.speaker,
            "reference_audio_id": req.reference_audio_id,
            "instruction": req.instruction,
            "speed": req.speed,
            "format": format_normalized,
            "stream": req.stream,
        },
        duration_ms=duration_ms,
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/indextts2")
def generate_indextts2(req: IndexTTS2Request):
    format_normalized = req.format.lower()
    output_id = f"aud_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            indextts2_runner.generate,
            text=req.text,
            output_path=str(output_path),
            speaker=req.speaker,
            reference_audio_id=req.reference_audio_id,
            emotion=req.emotion,
            duration_control=req.duration_control,
            speed=req.speed,
            format=format_normalized,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    return _audio_response(
        model_id="indextts-2",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "speaker": req.speaker,
            "reference_audio_id": req.reference_audio_id,
            "emotion": req.emotion,
            "duration_control": req.duration_control,
            "speed": req.speed,
            "format": format_normalized,
        },
        duration_ms=duration_ms,
        audio_meta=audio_meta,
    )


@app.post("/generate/audio/stable-audio-open")
def generate_stable_audio_open(req: StableAudioOpenRequest):
    format_normalized = req.format.lower()
    output_id = f"aud_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            stable_audio_open_runner.generate,
            prompt=req.prompt,
            output_path=str(output_path),
            duration_seconds=req.duration_seconds,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            random_seed=req.random_seed,
            format=format_normalized,
            negative_prompt=req.negative_prompt,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    return _audio_response(
        model_id="stable-audio-open-1.0",
        audio_kind="text-to-audio",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "duration_seconds": req.duration_seconds,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "format": format_normalized,
        },
        duration_ms=duration_ms,
        audio_meta=audio_meta,
    )
