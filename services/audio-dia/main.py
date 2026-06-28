from __future__ import annotations

import uuid
from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import AUDIO_OUTPUT_DIR, audio_response, create_audio_family_app, ensure_output_exists
from runners.text_to_speech.dia_runner import DiaRunner

SERVICE_NAME = "audio-dia"
FAMILY_NAME = "audio-dia"
TTS_TEXT_MAX_LENGTH = 12000
PATH_MAX_LENGTH = 500
WAV_ONLY_PATTERN = "^(wav)$"

dia_runner = DiaRunner()

SUPPORTED_MODELS = {
    "dia-1-6b": {
        "display_name": "Dia 1.6B Dialogue TTS",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "audio_prompt_path": {"type": "string", "required": False, "max_length": PATH_MAX_LENGTH},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class DiaParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    audio_prompt_path: str | None = Field(default=None, max_length=PATH_MAX_LENGTH)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


DiaParams.model_rebuild()


def _generate_dia(**params) -> dict:
    req = DiaParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"dia_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    audio_meta = dia_runner.generate(
        text=req.text,
        output_path=str(output_path),
        audio_prompt_path=req.audio_prompt_path,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="dia-1-6b",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={"text": req.text, "format": format_normalized},
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={"dia-1-6b": _generate_dia},
)
