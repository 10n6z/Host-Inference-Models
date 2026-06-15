"""Video generation models: wan-1.3b, cogvideox-2b, ltx-video.

To add a video model: add a request schema, instantiate its runner, append an
entry to `model_registry()`, and add a `@router.post(...)` endpoint.
"""
from __future__ import annotations

import mimetypes
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import Field, model_validator

from config import (
    MAX_SEED,
    PROMPT_MAX_LENGTH,
    VIDEO_MAX_FRAMES,
    VIDEO_MAX_SIZE,
    VIDEO_MIN_SIZE,
    VIDEO_OUTPUT_DIR,
    VIDEO_SIZE_STEP,
    _output_url,
    _public_output_url,
    _resolve_seed,
    _run_with_timeout,
)
from common import (
    StrictRequestModel,
    _check_output_exists,
    _field_spec,
    _map_runtime_error,
    _utc_now_iso,
)

from runners.text_to_video.cogvideox_runner import CogVideoXRunner
from runners.text_to_video.ltx_video_runner import LTXVideoRunner
from runners.text_to_video.wan_t2v_runner import WanT2VRunner

router = APIRouter()

wan_t2v_runner = WanT2VRunner()
cogvideox_runner = CogVideoXRunner()
ltx_video_runner = LTXVideoRunner()


class VideoSeedMixin(StrictRequestModel):
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    random_seed: bool = True

    @model_validator(mode="after")
    def validate_seed_requirements(self):
        if not self.random_seed and self.seed is None:
            raise ValueError("seed is required when random_seed is false.")
        return self


class VideoRequestBase(VideoSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(768, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(432, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(49, ge=8, le=VIDEO_MAX_FRAMES)
    num_inference_steps: int = Field(30, ge=1, le=80)
    guidance_scale: float = Field(5.0, ge=0.0, le=25.0)


class WanT2VRequest(VideoRequestBase):
    num_frames: int = Field(49, ge=8, le=81)
    num_inference_steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)


class CogVideoXRequest(VideoRequestBase):
    num_frames: int = Field(49, ge=8, le=49)
    num_inference_steps: int = Field(50, ge=1, le=80)
    guidance_scale: float = Field(6.0, ge=0.0, le=20.0)


class LTXVideoRequest(VideoRequestBase):
    num_frames: int = Field(97, ge=8, le=257)
    num_inference_steps: int = Field(30, ge=1, le=80)
    guidance_scale: float = Field(3.0, ge=0.0, le=25.0)
    decode_timestep: float = Field(0.03, ge=0.0, le=1.0)
    decode_noise_scale: float = Field(0.025, ge=0.0, le=1.0)


def _video_response(model_id: str, output_id: str, output_path, parameters_used: dict[str, Any], duration_ms: int) -> dict[str, Any]:
    relative_url = _output_url("videos", output_id)
    return {
        "success": True,
        "model_id": model_id,
        "modality": "video",
        "output_url": relative_url,
        "public_output_url": _public_output_url(relative_url),
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "video/mp4",
        "parameters_used": parameters_used,
        "duration_ms": duration_ms,
        "created_at": _utc_now_iso(),
    }


def model_registry() -> list[dict[str, Any]]:
    return [
        {
            "id": "wan21-t2v-1.3b",
            "displayName": "Wan2.1 T2V 1.3B",
            "modality": "video",
            "endpoint": "/generate/video/wan-1.3b",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=768, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "height": _field_spec("integer", default=432, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "num_frames": _field_spec("integer", default=49, minimum=8, maximum=81),
                "num_inference_steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=5.0, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
            },
            "capabilities": {
                "fps": 16,
                "negativePrompt": True,
                "seed": True,
            },
        },
        {
            "id": "cogvideox-2b",
            "displayName": "CogVideoX 2B",
            "modality": "video",
            "endpoint": "/generate/video/cogvideox-2b",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=768, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "height": _field_spec("integer", default=432, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "num_frames": _field_spec("integer", default=49, minimum=8, maximum=49),
                "num_inference_steps": _field_spec("integer", default=50, minimum=1, maximum=80),
                "guidance_scale": _field_spec("number", default=6.0, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
            },
            "capabilities": {
                "fps": 8,
                "negativePrompt": True,
                "seed": True,
            },
        },
        {
            "id": "ltx-video-2b",
            "displayName": "LTX-Video 2B",
            "modality": "video",
            "endpoint": "/generate/video/ltx-video",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=768, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "height": _field_spec("integer", default=432, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "num_frames": _field_spec("integer", default=97, minimum=8, maximum=257),
                "num_inference_steps": _field_spec("integer", default=30, minimum=1, maximum=80),
                "guidance_scale": _field_spec("number", default=3.0, minimum=0.0, maximum=25.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "decode_timestep": _field_spec("number", default=0.03, minimum=0.0, maximum=1.0),
                "decode_noise_scale": _field_spec("number", default=0.025, minimum=0.0, maximum=1.0),
            },
            "capabilities": {
                "fps": 24,
                "negativePrompt": True,
                "seed": True,
            },
        },
    ]


@router.post("/generate/video/wan-1.3b")
def generate_wan_t2v(req: WanT2VRequest):
    output_id = f"vid_{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            wan_t2v_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            fps=16,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _video_response(
        "wan21-t2v-1.3b",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "num_frames": req.num_frames,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "fps": 16,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/video/cogvideox-2b")
def generate_cogvideox(req: CogVideoXRequest):
    output_id = f"vid_{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            cogvideox_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            fps=8,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _video_response(
        "cogvideox-2b",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "num_frames": req.num_frames,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "fps": 8,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/video/ltx-video")
def generate_ltx_video(req: LTXVideoRequest):
    output_id = f"vid_{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            ltx_video_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            fps=24,
            decode_timestep=req.decode_timestep,
            decode_noise_scale=req.decode_noise_scale,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _video_response(
        "ltx-video-2b",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "num_frames": req.num_frames,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "fps": 24,
            "decode_timestep": req.decode_timestep,
            "decode_noise_scale": req.decode_noise_scale,
        },
        int((time.perf_counter() - start) * 1000),
    )
