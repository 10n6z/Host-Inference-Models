from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.outetts_runner import OuteTTSRunner

SERVICE_NAME = "audio-outetts"
FAMILY_NAME = "audio-outetts"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

outetts_runners = {
    "0.2-500M": OuteTTSRunner(variant="0.2-500M"),
    "0.3-1B": OuteTTSRunner(variant="0.3-1B"),
}

SUPPORTED_MODELS = {
    "outetts": {
        "display_name": "OuteTTS (0.2-500M / 0.3-1B)",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "variant": {"type": "string", "default": "0.2-500M", "enum": ["0.2-500M", "0.3-1B"]},
            "temperature": {"type": "number", "default": 0.1, "min": 0.0, "max": 2.0},
            "repetition_penalty": {"type": "number", "default": 1.1, "min": 1.0, "max": 3.0},
            "max_length": {"type": "integer", "default": 4096, "min": 256, "max": 8192},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class OuteTTSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    variant: str = Field("0.2-500M", pattern="^(0\\.2-500M|0\\.3-1B)$")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=3.0)
    max_length: int = Field(4096, ge=256, le=8192)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_outetts(**params) -> dict:
    req = OuteTTSParams.model_validate(params)
    if req.variant not in outetts_runners:
        raise ValueError(f"Unsupported OuteTTS variant '{req.variant}'.")

    format_normalized = req.format.lower()
    output_id = f"outetts_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = outetts_runners[req.variant].generate(
        text=req.text,
        output_path=str(output_path),
        temperature=req.temperature,
        repetition_penalty=req.repetition_penalty,
        max_length=req.max_length,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id=f"outetts-{req.variant}",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "variant": req.variant,
            "temperature": req.temperature,
            "repetition_penalty": req.repetition_penalty,
            "max_length": req.max_length,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={"outetts": _generate_outetts},
)
