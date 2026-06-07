from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.chatterbox_runner import ChatterboxRunner

SERVICE_NAME = "audio-chatterbox"
FAMILY_NAME = "audio-chatterbox"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

chatterbox_runner = ChatterboxRunner()

SUPPORTED_MODELS = {
    "chatterbox": {
        "display_name": "Chatterbox TTS",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "exaggeration": {"type": "number", "default": 0.5, "min": 0.0, "max": 2.0},
            "cfg_weight": {"type": "number", "default": 0.5, "min": 0.0, "max": 2.0},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class ChatterboxParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_chatterbox(**params) -> dict:
    req = ChatterboxParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"chatterbox_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = chatterbox_runner.generate(
        text=req.text,
        output_path=str(output_path),
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="chatterbox",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "exaggeration": req.exaggeration,
            "cfg_weight": req.cfg_weight,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={"chatterbox": _generate_chatterbox},
)
