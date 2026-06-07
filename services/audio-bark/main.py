from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.bark_runner import BarkRunner

SERVICE_NAME = "audio-bark"
FAMILY_NAME = "audio-bark"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

bark_runner = BarkRunner(variant="small")
bark_full_runner = BarkRunner(variant="full")

SUPPORTED_MODELS = {
    "bark-small": {
        "display_name": "Bark Small",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "voice_preset": {"type": "string", "default": "v2/en_speaker_6"},
            "do_sample": {"type": "boolean", "default": True},
            "temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0},
            "semantic_temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0},
            "coarse_temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0},
            "fine_temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0},
            "semantic_max_new_tokens": {"type": "integer", "default": 768, "min": 1, "max": 2048},
            "coarse_max_new_tokens": {"type": "integer", "default": 1536, "min": 1, "max": 4096},
            "fine_max_new_tokens": {"type": "integer", "default": 1536, "min": 1, "max": 4096},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
    "bark-full": {
        "display_name": "Bark Full",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "voice_preset": {"type": "string", "default": "v2/en_speaker_6"},
            "do_sample": {"type": "boolean", "default": True},
            "temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class BarkSmallParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    voice_preset: str = Field("v2/en_speaker_6", min_length=1, max_length=100)
    do_sample: bool = Field(True)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    semantic_temperature: float = Field(1.0, ge=0.0, le=2.0)
    coarse_temperature: float = Field(1.0, ge=0.0, le=2.0)
    fine_temperature: float = Field(1.0, ge=0.0, le=2.0)
    semantic_max_new_tokens: int = Field(768, ge=1, le=2048)
    coarse_max_new_tokens: int = Field(1536, ge=1, le=4096)
    fine_max_new_tokens: int = Field(1536, ge=1, le=4096)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


class BarkFullParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    voice_preset: str = Field("v2/en_speaker_6", min_length=1, max_length=100)
    do_sample: bool = Field(True)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_bark_small(**params) -> dict:
    req = BarkSmallParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"bark_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = bark_runner.generate(
        text=req.text,
        output_path=str(output_path),
        voice_preset=req.voice_preset,
        do_sample=req.do_sample,
        temperature=req.temperature,
        semantic_temperature=req.semantic_temperature,
        coarse_temperature=req.coarse_temperature,
        fine_temperature=req.fine_temperature,
        semantic_max_new_tokens=req.semantic_max_new_tokens,
        coarse_max_new_tokens=req.coarse_max_new_tokens,
        fine_max_new_tokens=req.fine_max_new_tokens,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="bark-small",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "voice_preset": req.voice_preset,
            "do_sample": req.do_sample,
            "temperature": req.temperature,
            "semantic_temperature": req.semantic_temperature,
            "coarse_temperature": req.coarse_temperature,
            "fine_temperature": req.fine_temperature,
            "semantic_max_new_tokens": req.semantic_max_new_tokens,
            "coarse_max_new_tokens": req.coarse_max_new_tokens,
            "fine_max_new_tokens": req.fine_max_new_tokens,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


def _generate_bark_full(**params) -> dict:
    req = BarkFullParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"bark_full_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = bark_full_runner.generate(
        text=req.text,
        output_path=str(output_path),
        voice_preset=req.voice_preset,
        do_sample=req.do_sample,
        temperature=req.temperature,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="bark-full",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "voice_preset": req.voice_preset,
            "do_sample": req.do_sample,
            "temperature": req.temperature,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={
        "bark-small": _generate_bark_small,
        "bark-full": _generate_bark_full,
    },
)
