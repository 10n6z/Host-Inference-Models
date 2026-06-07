from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.kokoro_onnx_runner import KokoroONNXRunner
from runners.text_to_speech.kokoro_runner import KokoroRunner

SERVICE_NAME = "audio-kokoro"
FAMILY_NAME = "audio-kokoro"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

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

kokoro_runner = KokoroRunner()
kokoro_onnx_runner = KokoroONNXRunner()

SUPPORTED_MODELS = {
    "kokoro-82m": {
        "display_name": "Kokoro-82M",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "voice": {"type": "string", "default": "af_heart"},
            "language": {"type": "string", "default": "en"},
            "speed": {"type": "number", "default": 1.0, "min": 0.5, "max": 2.0},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
            "sample_rate": {"type": "integer", "default": 24000, "min": 24000, "max": 24000},
            "lang_code": {"type": "string", "required": False, "max_length": 1},
        },
    },
    "kokoro-82m-onnx": {
        "display_name": "Kokoro-82M ONNX",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "voice": {"type": "string", "default": "af_heart"},
            "speed": {"type": "number", "default": 1.0, "min": 0.1, "max": 5.0},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class KokoroParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    voice: str = Field("af_heart", min_length=1, max_length=100)
    language: str = Field("en", min_length=2, max_length=10)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    sample_rate: int = Field(24000, ge=24000, le=24000)
    lang_code: Optional[str] = Field(default=None, min_length=1, max_length=1)


class KokoroONNXParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    voice: str = Field("af_heart", min_length=1, max_length=100)
    speed: float = Field(1.0, ge=0.1, le=5.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


KokoroParams.model_rebuild()
KokoroONNXParams.model_rebuild()


def _generate_kokoro(**params) -> dict:
    req = KokoroParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"tts_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    language = req.language.lower()
    lang_code = req.lang_code or KOKORO_LANGUAGE_TO_CODE.get(language)
    if lang_code is None:
        raise ValueError(f"Unsupported language '{req.language}'.")

    audio_meta = kokoro_runner.generate(
        text=req.text,
        output_path=str(output_path),
        voice=req.voice,
        speed=req.speed,
        lang_code=lang_code,
        sample_rate=req.sample_rate,
    )
    ensure_output_exists(output_path)
    return audio_response(
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
        audio_meta=audio_meta,
    )


def _generate_kokoro_onnx(**params) -> dict:
    req = KokoroONNXParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"kokoro_onnx_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = kokoro_onnx_runner.generate(
        text=req.text,
        output_path=str(output_path),
        voice=req.voice,
        speed=req.speed,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="kokoro-82m-onnx",
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
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={
        "kokoro-82m": _generate_kokoro,
        "kokoro-82m-onnx": _generate_kokoro_onnx,
    },
)
