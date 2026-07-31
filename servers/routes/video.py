"""Video generation models: wan-1.3b, cogvideox-2b, ltx-video.

To add a video model: add a request schema, instantiate its runner, append an
entry to `model_registry()`, and add a `@router.post(...)` endpoint.
"""
from __future__ import annotations

from functools import partial
import base64
import binascii
import io
import mimetypes
import os
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
    _run_with_timeout as _run_with_timeout_base,
)
from common import (
    APIError,
    StrictRequestModel,
    _check_output_exists,
    _field_spec,
    _map_runtime_error,
    _utc_now_iso,
)

from runners.text_to_video.animatediff_runner import AnimateDiffV3Runner
from runners.text_to_video.cogvideox5b_runner import CogVideoX5BRunner
from runners.text_to_video.cogvideox_runner import CogVideoXRunner
from runners.text_to_video.hunyuanvideo_runner import HunyuanVideoRunner
from runners.text_to_video.ltx_video_expanded_runner import LTXVideo13BRunner, LTXVideoDistilledRunner
from runners.text_to_video.ltx_video_runner import LTXVideoRunner
from runners.text_to_video.mochi1_runner import Mochi1Runner
from runners.text_to_video.cogvideox15_runner import CogVideoX15Runner
from runners.text_to_video.wan_expanded_runner import Wan13DiffusersRunner, Wan22T2VRunner
from runners.text_to_video.wan_i2v_runner import WanFLF2V14BRunner, WanI2V14BRunner
from runners.text_to_video.zeroscope_runner import ZeroscopeRunner
from runners.text_to_video.skyreels_v2_df_13b_runner import SkyReelsV2DF13BRunner
from runners.text_to_video.wan_t2v_14b_runner import WanT2V14BRunner
from runners.text_to_video.wan_t2v_runner import WanT2VRunner

router = APIRouter()
_run_with_timeout = partial(
    _run_with_timeout_base,
    gpu_uuid_env="VIDEO_GPU_UUID" if os.getenv("VIDEO_GPU_UUID") else None,
)

wan_t2v_runner = WanT2VRunner()
wan_t2v_14b_runner = WanT2V14BRunner()
skyreels_v2_df_13b_runner = SkyReelsV2DF13BRunner()
cogvideox_runner = CogVideoXRunner()
ltx_video_runner = LTXVideoRunner()
cogvideox5b_runner = CogVideoX5BRunner()
mochi1_runner = Mochi1Runner()
hunyuanvideo_runner = HunyuanVideoRunner()
ltx_video_13b_runner = LTXVideo13BRunner()
ltx_video_distilled_runner = LTXVideoDistilledRunner()
cogvideox15_runner = CogVideoX15Runner()
wan13_diffusers_runner = Wan13DiffusersRunner()
wan22_t2v_runner = Wan22T2VRunner()
zeroscope_runner = ZeroscopeRunner()
wan_i2v_14b_runner = WanI2V14BRunner()
wan_flf2v_14b_runner = WanFLF2V14BRunner()
animatediff_v3_runner = AnimateDiffV3Runner()


def _decode_input_image(raw: str):
    from PIL import Image

    payload = raw.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise APIError("VALIDATION_ERROR", "image: invalid base64 data.", 422) from exc
    if not data:
        raise APIError("VALIDATION_ERROR", "image: decoded image is empty.", 422)
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise APIError("VALIDATION_ERROR", "image: could not decode image bytes.", 422) from exc


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


class WanT2V14BRequest(VideoRequestBase):
    width: int = Field(832, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(480, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(81, ge=8, le=81)
    num_inference_steps: int = Field(50, ge=1, le=60)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    shift: float = Field(5.0, ge=0.0, le=12.0)


class SkyReelsV2DF13BRequest(VideoRequestBase):
    width: int = Field(960, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(544, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(97, ge=8, le=VIDEO_MAX_FRAMES)
    num_inference_steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(6.0, ge=0.0, le=20.0)
    shift: float = Field(8.0, ge=0.0, le=12.0)
    base_num_frames: int = Field(97, ge=8, le=121)
    ar_step: int = Field(0, ge=0, le=20)
    causal_block_size: int = Field(1, ge=1, le=10)
    overlap_history: Optional[int] = Field(default=None, ge=0, le=64)
    addnoise_condition: float = Field(20.0, ge=0.0, le=50.0)


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


class CogVideoX5BRequest(VideoRequestBase):
    width: int = Field(720, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(480, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(49, ge=8, le=49)
    num_inference_steps: int = Field(50, ge=1, le=80)
    guidance_scale: float = Field(6.0, ge=0.0, le=20.0)


class Mochi1Request(VideoRequestBase):
    width: int = Field(848, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(480, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(84, ge=8, le=163)
    num_inference_steps: int = Field(64, ge=1, le=100)
    guidance_scale: float = Field(4.5, ge=0.0, le=20.0)


class HunyuanVideoRequest(VideoRequestBase):
    width: int = Field(720, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(1280, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(129, ge=8, le=129)
    num_inference_steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(6.0, ge=0.0, le=20.0)


class LTXVideo13BRequest(LTXVideoRequest):
    width: int = Field(768, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(512, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(121, ge=8, le=257)


class LTXVideoDistilledRequest(LTXVideoRequest):
    width: int = Field(768, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(512, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(97, ge=8, le=257)
    num_inference_steps: int = Field(8, ge=1, le=40)
    guidance_scale: float = Field(1.0, ge=0.0, le=25.0)


class CogVideoX15Request(CogVideoX5BRequest):
    width: int = Field(1360, ge=VIDEO_MIN_SIZE, le=1360, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(768, ge=VIDEO_MIN_SIZE, le=1360, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(81, ge=8, le=161)


class Wan13DiffusersRequest(WanT2V14BRequest):
    pass


class Wan22T2VRequest(WanT2V14BRequest):
    width: int = Field(1280, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(720, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(81, ge=8, le=121)


class WanI2V14BRequest(VideoRequestBase):
    image: str = Field(..., min_length=1)
    width: int = Field(832, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(480, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(81, ge=8, le=81)
    num_inference_steps: int = Field(40, ge=1, le=60)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)


class WanFLF2V14BRequest(WanI2V14BRequest):
    last_image: str = Field(..., min_length=1)
    width: int = Field(1280, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(720, ge=VIDEO_MIN_SIZE, le=1280, multiple_of=VIDEO_SIZE_STEP)
    guidance_scale: float = Field(5.5, ge=0.0, le=20.0)


class AnimateDiffV3Request(VideoRequestBase):
    width: int = Field(512, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(512, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(16, ge=8, le=64)
    num_inference_steps: int = Field(25, ge=1, le=60)
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0)


class ZeroscopeRequest(VideoRequestBase):
    width: int = Field(576, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    height: int = Field(320, ge=VIDEO_MIN_SIZE, le=VIDEO_MAX_SIZE, multiple_of=VIDEO_SIZE_STEP)
    num_frames: int = Field(24, ge=8, le=64)
    num_inference_steps: int = Field(40, ge=1, le=80)
    guidance_scale: float = Field(9.0, ge=0.0, le=20.0)


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
            "id": "wan21-t2v-14b",
            "displayName": "Wan2.1 T2V 14B",
            "modality": "video",
            "endpoint": "/generate/video/wan-14b",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=832, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "height": _field_spec("integer", default=480, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "num_frames": _field_spec("integer", default=81, minimum=8, maximum=81),
                "num_inference_steps": _field_spec("integer", default=50, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=5.0, minimum=0.0, maximum=20.0),
                "shift": _field_spec("number", default=5.0, minimum=0.0, maximum=12.0),
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
            "id": "skyreels-v2-df-1.3b-540p",
            "displayName": "SkyReels V2 DF 1.3B 540P",
            "modality": "video",
            "endpoint": "/generate/video/skyreels-v2-df-1.3b-540p",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=960, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "height": _field_spec("integer", default=544, minimum=VIDEO_MIN_SIZE, maximum=VIDEO_MAX_SIZE, step=VIDEO_SIZE_STEP),
                "num_frames": _field_spec("integer", default=97, minimum=8, maximum=VIDEO_MAX_FRAMES),
                "base_num_frames": _field_spec("integer", default=97, minimum=8, maximum=121),
                "num_inference_steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=6.0, minimum=0.0, maximum=20.0),
                "shift": _field_spec("number", default=8.0, minimum=0.0, maximum=12.0),
                "ar_step": _field_spec("integer", default=0, minimum=0, maximum=20),
                "causal_block_size": _field_spec("integer", default=1, minimum=1, maximum=10),
                "overlap_history": _field_spec("integer", required=False, minimum=0, maximum=64),
                "addnoise_condition": _field_spec("number", default=20.0, minimum=0.0, maximum=50.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
            },
            "capabilities": {
                "fps": 24,
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


@router.post("/generate/video/wan-14b")
def generate_wan_t2v_14b(req: WanT2V14BRequest):
    output_id = f"vid_{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            wan_t2v_14b_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            shift=req.shift,
            seed=seed_used,
            fps=16,
            timeout_seconds=1800,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _video_response(
        "wan21-t2v-14b",
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
            "shift": req.shift,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "fps": 16,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/video/skyreels-v2-df-1.3b-540p")
def generate_skyreels_v2_df_13b(req: SkyReelsV2DF13BRequest):
    output_id = f"vid_{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            skyreels_v2_df_13b_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            shift=req.shift,
            seed=seed_used,
            fps=24,
            base_num_frames=req.base_num_frames,
            ar_step=req.ar_step,
            causal_block_size=req.causal_block_size,
            overlap_history=req.overlap_history,
            addnoise_condition=req.addnoise_condition,
            timeout_seconds=1800,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _video_response(
        "skyreels-v2-df-1.3b-540p",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "num_frames": req.num_frames,
            "base_num_frames": req.base_num_frames,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "shift": req.shift,
            "ar_step": req.ar_step,
            "causal_block_size": req.causal_block_size,
            "overlap_history": req.overlap_history,
            "addnoise_condition": req.addnoise_condition,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "fps": 24,
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


def _video_generate(runner, model_id: str, req, fps: int = 8, timeout_seconds: int = 1800):
    output_id = f"vid_{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            fps=fps,
            timeout_seconds=timeout_seconds,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _video_response(
        model_id,
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
            "fps": fps,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/video/cogvideox-5b")
def generate_cogvideox5b(req: CogVideoX5BRequest):
    return _video_generate(cogvideox5b_runner, "cogvideox-5b", req, fps=8)


@router.post("/generate/video/mochi-1")
def generate_mochi1(req: Mochi1Request):
    return _video_generate(mochi1_runner, "mochi-1-preview", req, fps=24)


@router.post("/generate/video/hunyuanvideo")
def generate_hunyuanvideo(req: HunyuanVideoRequest):
    return _video_generate(hunyuanvideo_runner, "hunyuanvideo-t2v", req, fps=24)


@router.post("/generate/video/ltx-video-13b")
def generate_ltx_video_13b(req: LTXVideo13BRequest):
    return _video_generate(ltx_video_13b_runner, "ltx-video-13b-097", req, fps=24)


@router.post("/generate/video/ltx-distilled")
def generate_ltx_video_distilled(req: LTXVideoDistilledRequest):
    return _video_generate(ltx_video_distilled_runner, "ltx-video-097-distilled", req, fps=24)


@router.post("/generate/video/cogvideox-1-5-5b")
def generate_cogvideox15(req: CogVideoX15Request):
    return _video_generate(cogvideox15_runner, "cogvideox-1-5-5b", req, fps=8)


@router.post("/generate/video/wan-1.3b-diffusers")
def generate_wan13_diffusers(req: Wan13DiffusersRequest):
    return _video_generate(wan13_diffusers_runner, "wan21-t2v-1.3b-diffusers", req, fps=16)


@router.post("/generate/video/wan22-t2v")
def generate_wan22_t2v(req: Wan22T2VRequest):
    return _video_generate(wan22_t2v_runner, "wan22-t2v-a14b", req, fps=16)


def _video_i2v_generate(runner, model_id: str, req, fps: int = 16, timeout_seconds: int = 1800):
    output_id = f"vid_{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    input_image = _decode_input_image(req.image)
    last_image = _decode_input_image(req.last_image) if hasattr(req, "last_image") else None
    start = time.perf_counter()

    try:
        _run_with_timeout(
            runner.generate,
            prompt=req.prompt,
            image=input_image,
            last_image=last_image,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            fps=fps,
            timeout_seconds=timeout_seconds,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _video_response(
        model_id,
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
            "fps": fps,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/video/wan-i2v-14b")
def generate_wan_i2v_14b(req: WanI2V14BRequest):
    return _video_i2v_generate(wan_i2v_14b_runner, "wan21-i2v-14b-480p", req, fps=16)


@router.post("/generate/video/wan-flf2v-14b")
def generate_wan_flf2v_14b(req: WanFLF2V14BRequest):
    return _video_i2v_generate(wan_flf2v_14b_runner, "wan21-flf2v-14b-720p", req, fps=16)


@router.post("/generate/video/animatediff")
def generate_animatediff_v3(req: AnimateDiffV3Request):
    return _video_generate(animatediff_v3_runner, "animatediff-v3", req, fps=8)


@router.post("/generate/video/zeroscope")
def generate_zeroscope(req: ZeroscopeRequest):
    return _video_generate(zeroscope_runner, "zeroscope-v2-576w", req, fps=8)
