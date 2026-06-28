from __future__ import annotations

import uuid
from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import AUDIO_OUTPUT_DIR, audio_response, create_audio_family_app, ensure_output_exists
from runners.text_to_speech.cosyvoice2_runner import CosyVoice2Runner

SERVICE_NAME = "audio-cosyvoice"
FAMILY_NAME = "audio-cosyvoice"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

cosyvoice2_runner = CosyVoice2Runner()

SUPPORTED_MODELS = {
    "cosyvoice2-0.5b": {
        "display_name": "CosyVoice2 0.5B",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class CosyVoice2Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


CosyVoice2Params.model_rebuild()


def _generate(**params) -> dict:
    req = CosyVoice2Params.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"cosyvoice2_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id
    audio_meta = cosyvoice2_runner.generate(text=req.text, output_path=str(output_path))
    ensure_output_exists(output_path)
    return audio_response(
        model_id="cosyvoice2-0.5b",
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
    handlers={"cosyvoice2-0.5b": _generate},
)
