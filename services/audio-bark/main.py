from __future__ import annotations

import uuid
from typing import Optional

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

# Bark follows the suno/bark model card: by default it runs
# model.generate(**inputs, do_sample=True) with the model's tuned config.
# `temperature` is an optional override applied to every Bark stage.
_BARK_FIELDS = {
    "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
    "voice_preset": {"type": "string", "default": "v2/en_speaker_6"},
    "do_sample": {"type": "boolean", "default": True},
    "temperature": {"type": "number", "min": 0.0, "max": 2.0},
    "seed": {"type": "integer", "min": 0, "max": 4294967295},
    "format": {"type": "string", "default": "wav", "enum": ["wav"]},
}

SUPPORTED_MODELS = {
    "bark-small": {
        "display_name": "Bark Small",
        "modality": "text-to-speech",
        "fields": _BARK_FIELDS,
    },
    "bark-full": {
        "display_name": "Bark Full",
        "modality": "text-to-speech",
        "fields": _BARK_FIELDS,
    },
}


class BarkParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    # Empty string is allowed: it means Unconditional (no acoustic prompt).
    voice_preset: str = Field("v2/en_speaker_6", max_length=100)
    do_sample: bool = Field(True)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    # Optional fixed seed for reproducible output; omit for a random take.
    seed: Optional[int] = Field(None, ge=0, le=4294967295)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate(runner: BarkRunner, model_id: str, prefix: str, **params) -> dict:
    req = BarkParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"{prefix}_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = runner.generate(
        text=req.text,
        output_path=str(output_path),
        voice_preset=req.voice_preset,
        do_sample=req.do_sample,
        temperature=req.temperature,
        seed=req.seed,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id=model_id,
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "voice_preset": req.voice_preset,
            "do_sample": req.do_sample,
            "temperature": req.temperature,
            "seed": req.seed,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


def _generate_bark_small(**params) -> dict:
    return _generate(bark_runner, "bark-small", "bark", **params)


def _generate_bark_full(**params) -> dict:
    return _generate(bark_full_runner, "bark-full", "bark_full", **params)


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={
        "bark-small": _generate_bark_small,
        "bark-full": _generate_bark_full,
    },
)
