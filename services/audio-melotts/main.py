from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.melotts_runner import (
    DEFAULT_SPEAKER_BY_LANGUAGE,
    SUPPORTED_LANGUAGES,
    MeloTTSRunner,
)

SERVICE_NAME = "audio-melotts"
FAMILY_NAME = "audio-melotts"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

melotts_runner = MeloTTSRunner()

SUPPORTED_MODELS = {
    "melotts": {
        "display_name": "MeloTTS",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "language": {
                "type": "string",
                "default": "EN",
                "enum": list(SUPPORTED_LANGUAGES),
            },
            "speaker": {"type": "string", "default": "EN-US"},
            "speed": {"type": "number", "default": 1.0, "min": 0.5, "max": 3.0},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class MeloTTSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    language: str = Field("EN")
    speaker: str | None = Field(None)
    speaker_id: int | None = Field(None, ge=0)
    speed: float = Field(1.0, ge=0.5, le=3.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_melotts(**params) -> dict:
    req = MeloTTSParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"melotts_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = melotts_runner.generate(
        text=req.text,
        output_path=str(output_path),
        language=req.language,
        speaker=req.speaker,
        speaker_id=req.speaker_id,
        speed=req.speed,
    )
    ensure_output_exists(output_path)
    used = audio_meta.get("parameters", {})
    return audio_response(
        model_id="melotts",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "language": used.get("language", req.language),
            "speaker": used.get("speaker", req.speaker),
            "speed": req.speed,
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


@app.get("/load_speakers")
def load_speakers(language: str = "EN"):
    """Return the named speakers for a language (mirrors Gradio /load_speakers)."""
    try:
        speakers = melotts_runner.list_speakers(language)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    normalized = language.strip().upper()
    return {
        "language": normalized,
        "speakers": speakers,
        "default_speaker": DEFAULT_SPEAKER_BY_LANGUAGE.get(normalized, speakers[0] if speakers else None),
    }
