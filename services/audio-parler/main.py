from __future__ import annotations

import uuid
from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import AUDIO_OUTPUT_DIR, audio_response, create_audio_family_app, ensure_output_exists
from runners.text_to_speech.parler_runner import ParlerLargeRunner, ParlerMiniRunner

SERVICE_NAME = "audio-parler"
FAMILY_NAME = "audio-parler"
TTS_TEXT_MAX_LENGTH = 12000
DESCRIPTION_MAX_LENGTH = 2000
WAV_ONLY_PATTERN = "^(wav)$"

parler_mini_runner = ParlerMiniRunner()
parler_large_runner = ParlerLargeRunner()

SUPPORTED_MODELS = {
    "parler-tts-mini-v1": {
        "display_name": "Parler-TTS Mini v1",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "description": {"type": "string", "required": False, "max_length": DESCRIPTION_MAX_LENGTH},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
    "parler-tts-large-v1": {
        "display_name": "Parler-TTS Large v1",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "description": {"type": "string", "required": False, "max_length": DESCRIPTION_MAX_LENGTH},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class ParlerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


ParlerParams.model_rebuild()


def _generate(model_id: str, runner, **params) -> dict:
    req = ParlerParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"parler_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    audio_meta = runner.generate(text=req.text, description=req.description, output_path=str(output_path))
    ensure_output_exists(output_path)
    return audio_response(
        model_id=model_id,
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={"text": req.text, "description": req.description, "format": format_normalized},
        audio_meta=audio_meta,
    )


def _generate_mini(**params) -> dict:
    return _generate("parler-tts-mini-v1", parler_mini_runner, **params)


def _generate_large(**params) -> dict:
    return _generate("parler-tts-large-v1", parler_large_runner, **params)


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={
        "parler-tts-mini-v1": _generate_mini,
        "parler-tts-large-v1": _generate_large,
    },
)
