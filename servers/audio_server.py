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

MODEL_CACHE_ROOT = Path(os.getenv("MODEL_CACHE_ROOT", BASE_DIR.parent / "model-cache")).resolve()
HF_HOME = Path(os.getenv("HF_HOME", MODEL_CACHE_ROOT / "huggingface")).resolve()
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE", HF_HOME / "hub")).resolve()
TORCH_HOME = Path(os.getenv("TORCH_HOME", MODEL_CACHE_ROOT / "torch")).resolve()
XDG_CACHE_HOME = Path(os.getenv("XDG_CACHE_HOME", MODEL_CACHE_ROOT / "cache")).resolve()
MPLCONFIGDIR = Path(os.getenv("MPLCONFIGDIR", XDG_CACHE_HOME / "matplotlib")).resolve()
HF_MODELS_ROOT = Path(os.getenv("HF_MODELS_ROOT", MODEL_CACHE_ROOT / "hf")).resolve()
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", BASE_DIR.parent / "outputs")).resolve()
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audio"
PUBLIC_BASE_URL = os.getenv("AUDIO_PUBLIC_BASE_URL", "http://localhost:8002").rstrip("/")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "300"))

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
TORCH_HOME.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
HF_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME))
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_HOME)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ["HF_MODELS_ROOT"] = str(HF_MODELS_ROOT)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from common import (
    APIError,
    StrictRequestModel,
    _check_output_exists,
    _field_spec,
    _map_runtime_error,
    _utc_now_iso,
    _validation_message,
)

from runners.text_to_audio.stable_audio_open import StableAudioOpenRunner
from runners.text_to_speech.e2_tts_runner import E2TTSRunner
from runners.text_to_speech.f5_tts_runner import F5TTSRunner
from runners.text_to_speech.kitten_tts_runner import KittenTTSRunner
from runners.text_to_speech.mms_tts_runner import MMSTTSRunner
from runners.text_to_speech.speecht5_runner import SpeechT5Runner

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

app = FastAPI(title="Local AI Audio Generation Server")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

stable_audio_runner = StableAudioOpenRunner()
mms_tts_runners = {lang: MMSTTSRunner(lang=lang) for lang in ("eng", "deu", "fra", "spa")}
speecht5_runner = SpeechT5Runner()
f5_tts_runner = F5TTSRunner()
e2_tts_runner = E2TTSRunner()
kitten_tts_runner = KittenTTSRunner()


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


class MMSTTSRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    language: str = Field("eng", pattern="^(eng|deu|fra|spa)$")
    speaking_rate: float = Field(1.0, ge=0.1, le=5.0)
    noise_scale: float = Field(0.667, ge=0.0, le=2.0)
    noise_scale_duration: float = Field(0.8, ge=0.0, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


class SpeechT5Request(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    speaker_index: int = Field(7306, ge=0, le=7930)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    minlenratio: float = Field(0.0, ge=0.0, le=10.0)
    maxlenratio: float = Field(20.0, ge=1.0, le=50.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


class F5TTSRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    ref_file: str = Field("", max_length=MAX_REF_AUDIO_PATH_LENGTH)
    ref_text: str = Field("", max_length=MAX_REFERENCE_TEXT_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=3.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


class E2TTSRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    ref_file: str = Field("", max_length=MAX_REF_AUDIO_PATH_LENGTH)
    ref_text: str = Field("", max_length=MAX_REFERENCE_TEXT_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=3.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


class KittenTTSRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    voice: str = Field("expr-voice-2-m", min_length=1, max_length=100)
    speed: float = Field(1.0, ge=0.5, le=3.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _model_registry() -> list[dict[str, Any]]:
    return [
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
        {
            "id": "mms-tts",
            "displayName": "MMS-TTS (eng/deu/fra/spa)",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/mms-tts",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "language": _field_spec("string", default="eng", enum=["eng", "deu", "fra", "spa"]),
                "speaking_rate": _field_spec("number", default=1.0, minimum=0.1, maximum=5.0),
                "noise_scale": _field_spec("number", default=0.667, minimum=0.0, maximum=2.0),
                "noise_scale_duration": _field_spec("number", default=0.8, minimum=0.0, maximum=2.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
        {
            "id": "speecht5-tts",
            "displayName": "SpeechT5 TTS",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/speecht5",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "speaker_index": _field_spec("integer", default=7306, minimum=0, maximum=7930),
                "threshold": _field_spec("number", default=0.5, minimum=0.0, maximum=1.0),
                "minlenratio": _field_spec("number", default=0.0, minimum=0.0, maximum=10.0),
                "maxlenratio": _field_spec("number", default=20.0, minimum=1.0, maximum=50.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
        {
            "id": "f5-tts",
            "displayName": "F5-TTS",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/f5-tts",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "ref_file": _field_spec("string", required=False, max_length=MAX_REF_AUDIO_PATH_LENGTH),
                "ref_text": _field_spec("string", required=False, max_length=MAX_REFERENCE_TEXT_LENGTH),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=3.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
        {
            "id": "e2-tts",
            "displayName": "E2-TTS",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/e2-tts",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "ref_file": _field_spec("string", required=False, max_length=MAX_REF_AUDIO_PATH_LENGTH),
                "ref_text": _field_spec("string", required=False, max_length=MAX_REFERENCE_TEXT_LENGTH),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=3.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
        {
            "id": "kitten-tts",
            "displayName": "KittenTTS Nano",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/kitten-tts",
            "fields": {
                "text": _field_spec("string", required=True, max_length=TTS_TEXT_MAX_LENGTH),
                "voice": _field_spec("string", default="expr-voice-2-m"),
                "speed": _field_spec("number", default=1.0, minimum=0.5, maximum=3.0),
                "format": _field_spec("string", default="wav", enum=["wav"]),
            },
        },
    ]


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


def _safe_error_message(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    if len(message) > MAX_ERROR_MESSAGE_LENGTH:
        return f"{message[:MAX_ERROR_MESSAGE_LENGTH]}..."
    return message


def _map_tts_runtime_error(exc: Exception) -> APIError:
    logger.exception("TTS generation failed")
    return APIError("TTS_GENERATION_FAILED", _safe_error_message(exc), 500)


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


@app.post("/generate/tts/mms-tts")
def generate_mms_tts(req: MMSTTSRequest):
    format_normalized = req.format.lower()
    lang = req.language.lower()
    output_id = f"mms_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    if lang not in mms_tts_runners:
        raise APIError(
            code="VALIDATION_ERROR",
            message=f"Unsupported MMS-TTS language '{lang}'. Supported: eng, deu, fra, spa.",
            status_code=422,
        )

    start = time.perf_counter()
    try:
        audio_meta = _run_with_timeout(
            mms_tts_runners[lang].generate,
            text=req.text,
            output_path=str(output_path),
            speaking_rate=req.speaking_rate,
            noise_scale=req.noise_scale,
            noise_scale_duration=req.noise_scale_duration,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_tts_runtime_error(exc)

    return _audio_response(
        model_id=f"mms-tts-{lang}",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "language": lang,
            "speaking_rate": req.speaking_rate,
            "noise_scale": req.noise_scale,
            "noise_scale_duration": req.noise_scale_duration,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/speecht5")
def generate_speecht5(req: SpeechT5Request):
    format_normalized = req.format.lower()
    output_id = f"speecht5_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            speecht5_runner.generate,
            text=req.text,
            output_path=str(output_path),
            speaker_index=req.speaker_index,
            threshold=req.threshold,
            minlenratio=req.minlenratio,
            maxlenratio=req.maxlenratio,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_tts_runtime_error(exc)

    return _audio_response(
        model_id="speecht5-tts",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "speaker_index": req.speaker_index,
            "threshold": req.threshold,
            "minlenratio": req.minlenratio,
            "maxlenratio": req.maxlenratio,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/f5-tts")
def generate_f5_tts(req: F5TTSRequest):
    format_normalized = req.format.lower()
    output_id = f"f5tts_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            f5_tts_runner.generate,
            text=req.text,
            output_path=str(output_path),
            ref_file=req.ref_file,
            ref_text=req.ref_text,
            speed=req.speed,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_tts_runtime_error(exc)

    return _audio_response(
        model_id="f5-tts",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "ref_file": req.ref_file,
            "ref_text": req.ref_text,
            "speed": req.speed,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/e2-tts")
def generate_e2_tts(req: E2TTSRequest):
    format_normalized = req.format.lower()
    output_id = f"e2tts_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            e2_tts_runner.generate,
            text=req.text,
            output_path=str(output_path),
            ref_file=req.ref_file,
            ref_text=req.ref_text,
            speed=req.speed,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_tts_runtime_error(exc)

    return _audio_response(
        model_id="e2-tts",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "ref_file": req.ref_file,
            "ref_text": req.ref_text,
            "speed": req.speed,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )


@app.post("/generate/tts/kitten-tts")
def generate_kitten_tts(req: KittenTTSRequest):
    format_normalized = req.format.lower()
    output_id = f"kitten_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    start = time.perf_counter()

    try:
        audio_meta = _run_with_timeout(
            kitten_tts_runner.generate,
            text=req.text,
            output_path=str(output_path),
            voice=req.voice,
            speed=req.speed,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_tts_runtime_error(exc)

    return _audio_response(
        model_id="kitten-tts",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "voice": req.voice,
            "speed": req.speed,
            "format": format_normalized,
        },
        duration_ms=int((time.perf_counter() - start) * 1000),
        audio_meta=audio_meta,
    )
