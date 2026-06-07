from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.melotts_runner import MeloTTSRunner

SERVICE_NAME = "audio-melotts"
FAMILY_NAME = "audio-melotts"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

melotts_runner = MeloTTSRunner()

SUPPORTED_MODELS = {
    "melotts": {
        "display_name": "MeloTTS English",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "speed": {"type": "number", "default": 1.0, "min": 0.5, "max": 3.0},
            "speaker_id": {"type": "integer", "default": 0, "min": 0, "max": 10},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class MeloTTSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=3.0)
    speaker_id: int = Field(0, ge=0, le=10)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_melotts(**params) -> dict:
    req = MeloTTSParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"melotts_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = melotts_runner.generate(
        text=req.text,
        output_path=str(output_path),
        speed=req.speed,
        speaker_id=req.speaker_id,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="melotts",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "speed": req.speed,
            "speaker_id": req.speaker_id,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={"melotts": _generate_melotts},
)
