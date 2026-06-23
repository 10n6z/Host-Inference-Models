from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from common.audio_service_base import (
    AUDIO_OUTPUT_DIR,
    create_audio_family_app,
    audio_response,
    ensure_output_exists,
)
from runners.text_to_speech.chatterbox_runner import ChatterboxRunner
from runners.text_to_speech.chatterbox_multilingual_runner import (
    ChatterboxMultilingualRunner,
    SUPPORTED_LANGUAGES,
    TEXT_MAX_LENGTH as MTL_TEXT_MAX_LENGTH,
)
from runners.text_to_speech.chatterbox_turbo_runner import (
    ChatterboxTurboRunner,
    EVENT_TAGS as TURBO_EVENT_TAGS,
    TEXT_MAX_LENGTH as TURBO_TEXT_MAX_LENGTH,
)

SERVICE_NAME = "audio-chatterbox"
FAMILY_NAME = "audio-chatterbox"
TTS_TEXT_MAX_LENGTH = 12000
WAV_ONLY_PATTERN = "^(wav)$"

chatterbox_runner = ChatterboxRunner()
chatterbox_mtl_runner = ChatterboxMultilingualRunner()
chatterbox_turbo_runner = ChatterboxTurboRunner()

SUPPORTED_MODELS = {
    "chatterbox": {
        "display_name": "Chatterbox TTS",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": TTS_TEXT_MAX_LENGTH},
            "audio_prompt_path": {"type": "string", "required": False},
            "exaggeration": {"type": "number", "default": 0.5, "min": 0.0, "max": 2.0},
            "cfg_weight": {"type": "number", "default": 0.5, "min": 0.0, "max": 2.0},
            "temperature": {"type": "number", "default": 0.8, "min": 0.05, "max": 5.0},
            "min_p": {"type": "number", "default": 0.05, "min": 0.0, "max": 1.0},
            "top_p": {"type": "number", "default": 1.0, "min": 0.0, "max": 1.0},
            "repetition_penalty": {"type": "number", "default": 1.2, "min": 1.0, "max": 2.0},
            "seed_num": {"type": "integer", "default": 0, "min": 0},
            "norm_loudness": {"type": "boolean", "default": False},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
    "chatterbox-multilingual": {
        "display_name": "Chatterbox Multilingual TTS V3",
        "modality": "text-to-speech",
        "fields": {
            "text": {"type": "string", "required": True, "max_length": MTL_TEXT_MAX_LENGTH},
            "language_id": {"type": "string", "default": "en", "enum": SUPPORTED_LANGUAGES},
            "audio_prompt_path": {"type": "string", "required": False},
            "exaggeration": {"type": "number", "default": 0.5, "min": 0.0, "max": 2.0},
            "temperature": {"type": "number", "default": 0.8, "min": 0.05, "max": 5.0},
            "seed_num": {"type": "integer", "default": 0, "min": 0},
            "cfg_weight": {"type": "number", "default": 0.5, "min": 0.0, "max": 2.0},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
    "chatterbox-turbo": {
        "display_name": "Chatterbox Turbo TTS",
        "modality": "text-to-speech",
        "fields": {
            "text": {
                "type": "string",
                "required": True,
                "max_length": TURBO_TEXT_MAX_LENGTH,
                "description": "Supports inline event tags: " + ", ".join(TURBO_EVENT_TAGS),
            },
            "audio_prompt_path": {"type": "string", "required": False},
            "temperature": {"type": "number", "default": 0.8, "min": 0.05, "max": 2.0},
            "seed_num": {"type": "integer", "default": 0, "min": 0},
            "min_p": {"type": "number", "default": 0.0, "min": 0.0, "max": 1.0},
            "top_p": {"type": "number", "default": 0.95, "min": 0.0, "max": 1.0},
            "top_k": {"type": "integer", "default": 1000, "min": 0, "max": 1000},
            "repetition_penalty": {"type": "number", "default": 1.2, "min": 1.0, "max": 2.0},
            "norm_loudness": {"type": "boolean", "default": True},
            "format": {"type": "string", "default": "wav", "enum": ["wav"]},
        },
    },
}


class ChatterboxParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TTS_TEXT_MAX_LENGTH)
    audio_prompt_path: str | None = Field(None)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=2.0)
    temperature: float = Field(0.8, ge=0.05, le=5.0)
    min_p: float = Field(0.05, ge=0.0, le=1.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0)
    seed_num: int = Field(0, ge=0)
    norm_loudness: bool = Field(False)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_chatterbox(**params) -> dict:
    req = ChatterboxParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"chatterbox_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = chatterbox_runner.generate(
        text=req.text,
        output_path=str(output_path),
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
        temperature=req.temperature,
        min_p=req.min_p,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        audio_prompt_path=req.audio_prompt_path,
        seed_num=req.seed_num,
        norm_loudness=req.norm_loudness,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="chatterbox",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "audio_prompt_path": req.audio_prompt_path,
            "exaggeration": req.exaggeration,
            "cfg_weight": req.cfg_weight,
            "temperature": req.temperature,
            "min_p": req.min_p,
            "top_p": req.top_p,
            "repetition_penalty": req.repetition_penalty,
            "seed_num": req.seed_num,
            "norm_loudness": req.norm_loudness,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


class ChatterboxMultilingualParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=MTL_TEXT_MAX_LENGTH)
    language_id: str = Field("en")
    audio_prompt_path: str | None = Field(None)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    temperature: float = Field(0.8, ge=0.05, le=5.0)
    seed_num: int = Field(0, ge=0)
    cfg_weight: float = Field(0.5, ge=0.0, le=2.0)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_chatterbox_multilingual(**params) -> dict:
    req = ChatterboxMultilingualParams.model_validate(params)
    if req.language_id not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"language_id: must be one of {', '.join(SUPPORTED_LANGUAGES)}"
        )
    format_normalized = req.format.lower()
    output_id = f"chatterbox_mtl_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = chatterbox_mtl_runner.generate(
        text=req.text,
        output_path=str(output_path),
        language_id=req.language_id,
        audio_prompt_path=req.audio_prompt_path,
        exaggeration=req.exaggeration,
        temperature=req.temperature,
        seed_num=req.seed_num,
        cfg_weight=req.cfg_weight,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="chatterbox-multilingual",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "language_id": req.language_id,
            "audio_prompt_path": req.audio_prompt_path,
            "exaggeration": req.exaggeration,
            "temperature": req.temperature,
            "seed_num": req.seed_num,
            "cfg_weight": req.cfg_weight,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


class ChatterboxTurboParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=TURBO_TEXT_MAX_LENGTH)
    audio_prompt_path: str | None = Field(None)
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    seed_num: int = Field(0, ge=0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(1000, ge=0, le=1000)
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0)
    norm_loudness: bool = Field(True)
    format: str = Field("wav", pattern=WAV_ONLY_PATTERN)


def _generate_chatterbox_turbo(**params) -> dict:
    req = ChatterboxTurboParams.model_validate(params)
    format_normalized = req.format.lower()
    output_id = f"chatterbox_turbo_{uuid.uuid4().hex}.{format_normalized}"
    output_path = AUDIO_OUTPUT_DIR / output_id

    audio_meta = chatterbox_turbo_runner.generate(
        text=req.text,
        output_path=str(output_path),
        audio_prompt_path=req.audio_prompt_path,
        temperature=req.temperature,
        seed_num=req.seed_num,
        min_p=req.min_p,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        norm_loudness=req.norm_loudness,
    )
    ensure_output_exists(output_path)
    return audio_response(
        model_id="chatterbox-turbo",
        modality="text-to-speech",
        audio_kind="tts",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
            "text": req.text,
            "audio_prompt_path": req.audio_prompt_path,
            "temperature": req.temperature,
            "seed_num": req.seed_num,
            "min_p": req.min_p,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
            "norm_loudness": req.norm_loudness,
            "format": format_normalized,
        },
        audio_meta=audio_meta,
    )


app = create_audio_family_app(
    service_name=SERVICE_NAME,
    family_name=FAMILY_NAME,
    supported_models=SUPPORTED_MODELS,
    handlers={
        "chatterbox": _generate_chatterbox,
        "chatterbox-multilingual": _generate_chatterbox_multilingual,
        "chatterbox-turbo": _generate_chatterbox_turbo,
    },
)
