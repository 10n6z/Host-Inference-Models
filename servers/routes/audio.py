"""Audio models served by the combined server: kokoro (TTS), stable-audio-open.

Standalone TTS models (mms, speecht5, f5, e2, kitten) live in the separate
audio-legacy service (audio_server.py), not here.

To add an audio model: add a request schema, instantiate its runner, append an
entry to `model_registry()`, and add a `@router.post(...)` endpoint.
"""
from __future__ import annotations

import mimetypes
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import Field, model_validator

from config import (
    AUDIO_OUTPUT_DIR,
    KOKORO_LANGUAGE_TO_CODE,
    MAX_SEED,
    STABLE_AUDIO_MAX_DURATION_SECONDS,
    STABLE_AUDIO_MIN_DURATION_SECONDS,
    TTA_TEXT_MAX_LENGTH,
    TTS_TEXT_MAX_LENGTH,
    WAV_ONLY_PATTERN,
    _output_url,
    _public_output_url,
    _resolve_seed,
    _run_with_timeout,
)
from common import (
    APIError,
    StrictRequestModel,
    _check_output_exists,
    _field_spec,
    _map_runtime_error,
    _utc_now_iso,
)

from runners.text_to_audio.stable_audio_open import StableAudioOpenRunner
from runners.text_to_speech.kokoro_runner import KokoroRunner

router = APIRouter()

kokoro_runner = KokoroRunner()
stable_audio_open_runner = StableAudioOpenRunner()


class KokoroRequest(StrictRequestModel):
    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    voice: str = Field("af_heart", min_length=1, max_length=100)
    language: str = Field("en", min_length=2, max_length=10)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    sample_rate: int = Field(24000, ge=24000, le=24000)
    lang_code: Optional[str] = Field(default=None, min_length=1, max_length=1)


class StableAudioOpenRequest(StrictRequestModel):
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    random_seed: bool = True
    prompt: str = Field(..., min_length=1, max_length=TTA_TEXT_MAX_LENGTH)
    duration_seconds: float = Field(10.0, ge=STABLE_AUDIO_MIN_DURATION_SECONDS, le=STABLE_AUDIO_MAX_DURATION_SECONDS)
    steps: int = Field(50, ge=1, le=250)
    guidance_scale: float = Field(7.0, ge=0.0, le=25.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    negative_prompt: Optional[str] = Field(default=None, max_length=TTA_TEXT_MAX_LENGTH)

    @model_validator(mode="after")
    def validate_seed_requirements(self):
        if not self.random_seed and self.seed is None:
            raise ValueError("seed is required when random_seed is false.")
        return self


def _audio_response(
    *,
    model_id: str,
    audio_kind: str,
    output_id: str,
    output_path,
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


def model_registry() -> list[dict[str, Any]]:
    return [
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


@router.post("/generate/tts/kokoro")
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


@router.post("/generate/audio/stable-audio-open")
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
