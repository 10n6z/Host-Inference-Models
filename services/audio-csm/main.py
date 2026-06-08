from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.csm_runner import CSMRunner

SERVICE_NAME = "audio-csm"
FAMILY_NAME = "audio-csm"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

csm_runner = CSMRunner()

SUPPORTED_MODELS = {
    "csm-1b": {
        "display_name": "Sesame CSM-1B",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "speaker": {"type": "integer", "default": 0, "min": 0, "max": 10},
            "max_audio_length_ms": {"type": "integer", "default": 10000, "min": 1000, "max": 30000},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class CSMParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    speaker: int = Field(0, ge=0, le=10)
    max_audio_length_ms: int = Field(10000, ge=1000, le=30000)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_csm(**params) -> dict:
    req = CSMParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"csm_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = csm_runner.generate(
        text=req.text,
        output_path=str(output_path),
        speaker=req.speaker,
        max_audio_length_ms=req.max_audio_length_ms,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="csm-1b",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "speaker": req.speaker,
            "max_audio_length_ms": req.max_audio_length_ms,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={"csm-1b": _generate_csm},
)
