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
from runners.text_to_speech.xtts_runner import XTTSRunner

SERVICE_NAME = "audio-xtts"
FAMILY_NAME = "audio-xtts"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

xtts_runner = XTTSRunner()

SUPPORTED_MODELS = {
    "xtts-v2": {
        "display_name": "XTTS v2",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "language": {"type": "string", "default": "en"},
            "speaker_wav": {"type": "string", "required": False, "max_length": 500},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class XTTSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    language: str = Field("en", min_length=2, max_length=10)
    speaker_wav: Optional[str] = Field(default=None, max_length=500)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_xtts(**params) -> dict:
    req = XTTSParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"xtts_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = xtts_runner.generate(
        text=req.text,
        output_path=str(output_path),
        language=req.language,
        speaker_wav=req.speaker_wav or "",
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="xtts-v2",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "language": req.language,
            "speaker_wav": req.speaker_wav,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={"xtts-v2": _generate_xtts},
)
