from __future__ import annotations

import mimetypes
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

HF_HOME = Path(os.getenv("HF_HOME", BASE_DIR.parent / "models" / "hf-cache")).resolve()
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE", HF_HOME / "hub")).resolve()
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", BASE_DIR.parent / "outputs")).resolve()
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audio"
PUBLIC_BASE_URL = os.getenv("AUDIO_PUBLIC_BASE_URL", "http://localhost:8002").rstrip("/")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "300"))

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from runners.text_to_audio.stable_audio_open import StableAudioOpenRunner
from runners.text_to_speech.cosyvoice2_runner import CosyVoice2Runner
from runners.text_to_speech.fish_speech_runner import FishSpeechRunner
from runners.text_to_speech.indextts2_runner import IndexTTS2Runner
from runners.text_to_speech.kokoro_runner import KokoroRunner

TTS_TEXT_MAX_LENGTH = 12000
TTA_TEXT_MAX_LENGTH = 4000
MAX_SEED = 2_147_483_647
MAX_REF_AUDIO_PATH_LENGTH = 500
MAX_REFERENCE_TEXT_LENGTH = 12000
STABLE_AUDIO_MIN_DURATION_SECONDS = 1.0
STABLE_AUDIO_MAX_DURATION_SECONDS = 47.0
WAV_ONLY_PATTERN = "^(wav)$"
MAX_ERROR_MESSAGE_LENGTH = 500

logger = logging.getLogger(__name__)

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

app = FastAPI(title="Local AI Audio Generation Server")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

kokoro_runner = KokoroRunner()
cosyvoice2_runner = CosyVoice2Runner()
fish_speech_runner = FishSpeechRunner()
indextts2_runner = IndexTTS2Runner()
stable_audio_runner = StableAudioOpenRunner()


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: Optional[dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KokoroRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    voice: str = Field("af_heart", min_length=1, max_length=100)
    language: str = Field("en", min_length=2, max_length=10)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    sample_rate: int = Field(24000, ge=24000, le=24000)
    lang_code: Optional[str] = Field(default=None, min_length=1, max_length=1)


class CosyVoice2Request(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    reference_audio: Optional[str] = Field(default=None, min_length=1, max_length=MAX_REF_AUDIO_PATH_LENGTH)
    reference_text: Optional[str] = Field(default=None, max_length=MAX_REFERENCE_TEXT_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    speaker: str = Field("default", min_length=1, max_length=100)
    instruction: Optional[str] = Field(default=None, max_length=500)
    stream: bool = False


class FishSpeechRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    reference_audio: Optional[str] = Field(default=None, min_length=1, max_length=MAX_REF_AUDIO_PATH_LENGTH)
    reference_text: Optional[str] = Field(default=None, max_length=MAX_REFERENCE_TEXT_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    voice: str = Field("default", min_length=1, max_length=100)
    sample_rate: Optional[int] = Field(default=None, ge=8000, le=96000)


class IndexTTS2Request(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    reference_audio: Optional[str] = Field(default=None, min_length=1, max_length=MAX_REF_AUDIO_PATH_LENGTH)
    reference_text: Optional[str] = Field(default=None, max_length=MAX_REFERENCE_TEXT_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    speaker: str = Field("default", min_length=1, max_length=100)
    emotion: Optional[str] = Field(default=None, min_length=1, max_length=100)
    duration_control: Optional[float] = Field(default=None, ge=0.5, le=120.0)


class StableAudioRequest(StrictRequestModel):
    prompt: str = Field(..., min_length=1, max_length=TTA_TEXT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=TTA_TEXT_MAX_LENGTH)
    duration_seconds: float = Field(
        10.0,
        ge=STABLE_AUDIO_MIN_DURATION_SECONDS,
        le=STABLE_AUDIO_MAX_DURATION_SECONDS,
    )
    steps: int = Field(100, ge=1, le=250)
    guidance_scale: float = Field(7.0, ge=0.0, le=25.0)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _field_spec(
    field_type: str,
    *,
    required: Optional[bool] = None,
    default: Any = None,
    minimum: Any = None,
    maximum: Any = None,
    enum: Optional[list[Any]] = None,
    max_length: Optional[int] = None,
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
    if enum is not None:
        data["enum"] = enum
    if max_length is not None:
        data["max_length"] = max_length
    return data


def _model_registry() -> list[dict[str, Any]]:
    return [
        {
            "id": "kokoro-82m",
            "displayName": "Kokoro-82M",
            "modality": "text-to-speech",
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
        },
        {
            "id": "fish-speech-v1.5",
            "displayName": "Fish Speech v1.5",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/fish-speech",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "reference_audio": _field_spec("string", required=False, max_length=MAX_REF_AUDIO_PATH_LENGTH),
                "reference_text": _field_spec("string", required=False, max_length=MAX_REFERENCE_TEXT_LENGTH),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=2.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
        {
            "id": "cosyvoice2-0.5b",
            "displayName": "CosyVoice2-0.5B",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/cosyvoice2",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "reference_audio": _field_spec("string", required=False, max_length=MAX_REF_AUDIO_PATH_LENGTH),
                "reference_text": _field_spec("string", required=False, max_length=MAX_REFERENCE_TEXT_LENGTH),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=2.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
        {
            "id": "indextts-2",
            "displayName": "IndexTTS-2",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/indextts2",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "reference_audio": _field_spec("string", required=False, max_length=MAX_REF_AUDIO_PATH_LENGTH),
                "reference_text": _field_spec("string", required=False, max_length=MAX_REFERENCE_TEXT_LENGTH),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=2.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
        {
            "id": "stable-audio-open-1.0",
            "displayName": "Stable Audio Open 1.0",
            "modality": "text-to-audio",
            "endpoint": "/generate/audio/stable-audio",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=TTA_TEXT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=TTA_TEXT_MAX_LENGTH),
                "duration_seconds": _field_spec(
                    "number",
                    default=10.0,
                    minimum=STABLE_AUDIO_MIN_DURATION_SECONDS,
                    maximum=STABLE_AUDIO_MAX_DURATION_SECONDS,
                ),
                "steps": _field_spec("integer", default=100, minimum=1, maximum=250),
                "guidance_scale": _field_spec("number", default=7.0, minimum=0.0, maximum=25.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
    ]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _output_url(file_name: str) -> str:
    return f"/outputs/audio/{file_name}"


def _public_output_url(relative_output_url: str) -> str:
    return f"{PUBLIC_BASE_URL}{relative_output_url}"


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


def _safe_error_message(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    if len(message) > MAX_ERROR_MESSAGE_LENGTH:
        return f"{message[:MAX_ERROR_MESSAGE_LENGTH]}..."
    return message


def _map_tts_runtime_error(exc: Exception) -> APIError:
    logger.exception("TTS generation failed")
    return APIError("TTS_GENERATION_FAILED", _safe_error_message(exc), 500)


def _validation_message(errors: list[dict[str, Any]]) -> str:
    first = errors[0] if errors else {}
    loc = first.get("loc", [])
    loc_text = ".".join(str(part) for part in loc if part not in ("body",))
    msg = first.get("msg", "Invalid request body.")
    if loc_text:
        return f"{loc_text}: {msg}"
    return msg


def _audio_response(
    *,
    model_id: str,
    modality: str,
    audio_kind: str,
    output_id: str,
    output_path: Path,
    parameters_used: dict[str, Any],
    duration_ms: int,
    audio_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    relative_url = _output_url(output_id)
    output_url = _public_output_url(relative_url)
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
        "status": "completed",
        "modelId": model_id,
        "model_id": model_id,
        "modality": modality,
        "audio_kind": audio_kind,
        "outputType": "audio",
        "outputId": output_id,
        "outputUrl": output_url,
        "output_url": relative_url,
        "public_output_url": output_url,
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "audio/wav",
        "parameters_used": parameters_used,
        "duration_ms": duration_ms,
        "audio_duration_seconds": audio_duration_seconds,
        "duration_seconds": audio_duration_seconds,
        "sample_rate": sample_rate,
        "created_at": _utc_now_iso(),
    }


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
        "service": "local-ai-audio-generation-server",
        "created_at": _utc_now_iso(),
    }


@app.get("/models")
def models():
    return {"models": _model_registry()}


@app.post("/generate/tts/kokoro")
def generate_kokoro(req: KokoroRequest):
    format_normalized = req.format.lower()
    output_id = f"tts_{uuid.uuid4().hex}.{format_normalized}"
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

    return _audio_response(
        model_id="kokoro-82m",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "voice": req.voice,
            "language": language,
            "lang_code": lang_code,
            "speed": req.speed,
            "format": format_normalized,
            "sample_rate": req.sample_rate,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/fish-speech")
def generate_fish_speech(req: FishSpeechRequest):
    format_normalized = req.format.lower()
    output_id = f"fish_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            fish_speech_runner.generate,
            text=req.text,
            output_path=str(output_path),
            voice=req.voice,
            speed=req.speed,
            format=format_normalized,
            seed=req.seed,
            sample_rate=req.sample_rate,
            reference_audio_id=req.reference_audio,
            parameters={
                "referenceText": req.reference_text,
                "seed": req.seed,
            },
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_tts_runtime_error(exc)

    return _audio_response(
        model_id="fish-speech-v1.5",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "seed": req.seed,
            "reference_audio": req.reference_audio,
            "reference_text": req.reference_text,
            "speed": req.speed,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/cosyvoice2")
def generate_cosyvoice2(req: CosyVoice2Request):
    format_normalized = req.format.lower()
    output_id = f"cosyvoice2_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            cosyvoice2_runner.generate,
            text=req.text,
            output_path=str(output_path),
            speaker=req.speaker,
            speed=req.speed,
            format=format_normalized,
            seed=req.seed,
            reference_audio_id=req.reference_audio,
            instruction=req.instruction,
            stream=req.stream,
            parameters={
                "referenceText": req.reference_text,
                "seed": req.seed,
            },
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_tts_runtime_error(exc)

    return _audio_response(
        model_id="cosyvoice2-0.5b",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "seed": req.seed,
            "reference_audio": req.reference_audio,
            "reference_text": req.reference_text,
            "speed": req.speed,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/indextts2")
def generate_indextts2(req: IndexTTS2Request):
    format_normalized = req.format.lower()
    output_id = f"indextts2_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            indextts2_runner.generate,
            text=req.text,
            output_path=str(output_path),
            speaker=req.speaker,
            speed=req.speed,
            format=format_normalized,
            seed=req.seed,
            reference_audio_id=req.reference_audio,
            emotion=req.emotion,
            duration_control=req.duration_control,
            parameters={
                "referenceText": req.reference_text,
                "seed": req.seed,
            },
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_tts_runtime_error(exc)

    return _audio_response(
        model_id="indextts-2",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "seed": req.seed,
            "reference_audio": req.reference_audio,
            "reference_text": req.reference_text,
            "speed": req.speed,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )


@app.post("/generate/audio/stable-audio")
def generate_stable_audio(req: StableAudioRequest):
    format_normalized = req.format.lower()
    output_id = f"audio_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            stable_audio_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            duration_seconds=req.duration_seconds,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            format=format_normalized,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _audio_response(
        model_id="stable-audio-open-1.0",
        modality="text-to-audio",
        audio_kind="text-to-audio",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "duration_seconds": req.duration_seconds,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": req.seed,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )
