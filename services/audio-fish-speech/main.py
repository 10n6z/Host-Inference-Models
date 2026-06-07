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
from runners.text_to_speech.fish_speech_runner import FishSpeechRunner

SERVICE_NAME = "audio-fish-speech"
FAMILY_NAME = "audio-fish-speech"
TTS_TEXT_MAX_LENGTH = 12000
MAX_SEED = 2_147_483_647
MAX_REF_AUDIO_PATH_LENGTH = 500
MAX_REFERENCE_TEXT_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

fish_speech_runner = FishSpeechRunner()

SUPPORTED_MODELS = {
    "fish-speech-v1.5": {
        "display_name": "Fish Speech v1.5",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "seed": {"type": "integer", "required": False, "min": 0, "max": MAX_SEED},
            "reference_audio": {"type": "string", "required": False, "max_length": MAX_REF_AUDIO_PATH_LENGTH},
            "reference_text": {"type": "string", "required": False, "max_length": MAX_REFERENCE_TEXT_LENGTH},
            "speed": {"type": "number", "default": 1.0, "min": 0.5, "max": 2.0},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class FishSpeechParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    reference_audio: Optional[str] = Field(default=None, min_length=1, max_length=MAX_REF_AUDIO_PATH_LENGTH)
    reference_text: Optional[str] = Field(default=None, max_length=MAX_REFERENCE_TEXT_LENGTH)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)
    voice: str = Field("default", min_length=1, max_length=100)
    sample_rate: Optional[int] = Field(default=None, ge=8000, le=96000)


def _generate_fish_speech(**params) -> dict:
    req = FishSpeechParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"fish_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = fish_speech_runner.generate(
        text=req.text,
        output_path=str(output_path),
        voice=req.voice,
        speed=req.speed,
        format=format_normalized,
        seed=req.seed,
        sample_rate=req.sample_rate,
        reference_audio_id=req.reference_audio,
        parameters={
            "referenceText": req.reference_text,
            "seed": req.seed,
        },
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="fish-speech-v1.5",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "seed": req.seed,
            "reference_audio": req.reference_audio,
            "reference_text": req.reference_text,
            "speed": req.speed,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={"fish-speech-v1.5": _generate_fish_speech},
)
