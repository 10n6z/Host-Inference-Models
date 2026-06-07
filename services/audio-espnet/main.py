from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.espnet_vits_runner import ESPnetVITSRunner

SERVICE_NAME = "audio-espnet"
FAMILY_NAME = "audio-espnet"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

espnet_vits_runner = ESPnetVITSRunner()

SUPPORTED_MODELS = {
    "espnet-vits": {
        "display_name": "ESPnet VITS (LJSpeech)",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "alpha": {"type": "number", "default": 1.0, "min": 0.1, "max": 3.0},
            "noise_scale": {"type": "number", "default": 0.667, "min": 0.0, "max": 2.0},
            "noise_scale_dur": {"type": "number", "default": 0.8, "min": 0.0, "max": 2.0},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class ESPnetVITSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    alpha: float = Field(1.0, ge=0.1, le=3.0)
    noise_scale: float = Field(0.667, ge=0.0, le=2.0)
    noise_scale_dur: float = Field(0.8, ge=0.0, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_espnet_vits(**params) -> dict:
    req = ESPnetVITSParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"espnet_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = espnet_vits_runner.generate(
        text=req.text,
        output_path=str(output_path),
        alpha=req.alpha,
        noise_scale=req.noise_scale,
        noise_scale_dur=req.noise_scale_dur,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="espnet-vits",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "alpha": req.alpha,
            "noise_scale": req.noise_scale,
            "noise_scale_dur": req.noise_scale_dur,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={"espnet-vits": _generate_espnet_vits},
)
