"""Image generation models: flux, sd35, auraflow, openflux.

To add an image model: add a request schema, instantiate its runner, append an
entry to `model_registry()`, and add a `@router.post(...)` endpoint.
"""
from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import Field, model_validator

from config import (
    IMAGE_MAX_SIZE,
    IMAGE_MIN_SIZE,
    IMAGE_OUTPUT_DIR,
    IMAGE_SIZE_STEP,
    MAX_NUM_IMAGES,
    MAX_SEED,
    PROMPT_MAX_LENGTH,
    _output_url,
    _public_output_url,
    _resolve_seed,
    _run_with_timeout,
)
from common import (
    APIError,
    StrictRequestModel,
    _check_output_exists,
    _field_spec,
    _map_runtime_error,
    _utc_now_iso,
)

from runners.text_to_image.auraflow import AuraFlowRunner
from runners.text_to_image.flux_schnell import FluxSchnellRunner
from runners.text_to_image.openflux import OpenFluxRunner
from runners.text_to_image.sd35_medium import SD35MediumRunner
from runners.image_to_image.qwen_image_edit import QwenImageEditRunner

router = APIRouter()

flux_runner = FluxSchnellRunner()
sd35_runner = SD35MediumRunner()
auraflow_runner = AuraFlowRunner()
openflux_runner = OpenFluxRunner()
qwen_image_edit_runner = QwenImageEditRunner()


def _decode_input_image(raw: str):
    """Decode a base64 (raw or data URL) string into an RGB PIL image."""
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
    except Exception as exc:  # noqa: BLE001
        raise APIError("VALIDATION_ERROR", "image: could not decode image bytes.", 422) from exc


class ImageSeedMixin(StrictRequestModel):
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    random_seed: bool = True

    @model_validator(mode="after")
    def validate_seed_requirements(self):
        if not self.random_seed and self.seed is None:
            raise ValueError("seed is required when random_seed is false.")
        return self


class FluxRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(4, ge=1, le=20)
    guidance_scale: float = Field(0.0, ge=0.0, le=0.0)
    max_sequence_length: int = Field(256, ge=1, le=256)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class SD35Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(28, ge=1, le=60)
    guidance_scale: float = Field(7.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class AuraFlowRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class OpenFluxRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(28, ge=1, le=60)
    guidance_scale: float = Field(7.0, ge=0.0, le=20.0)
    max_sequence_length: int = Field(512, ge=1, le=512)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class QwenImageEditRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    image: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    steps: int = Field(20, ge=1, le=60)
    true_cfg_scale: float = Field(4.0, ge=0.0, le=20.0)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


def _image_response(model_id: str, output_id: str, output_path, parameters_used: dict[str, Any], duration_ms: int) -> dict[str, Any]:
    relative_url = _output_url("images", output_id)
    return {
        "success": True,
        "model_id": model_id,
        "modality": "image",
        "output_url": relative_url,
        "public_output_url": _public_output_url(relative_url),
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "image/png",
        "parameters_used": parameters_used,
        "duration_ms": duration_ms,
        "created_at": _utc_now_iso(),
    }


def model_registry() -> list[dict[str, Any]]:
    return [
        {
            "id": "flux-1-schnell",
            "displayName": "FLUX.1 Schnell",
            "modality": "image",
            "endpoint": "/generate/image/flux",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=4, minimum=1, maximum=20),
                "guidance_scale": _field_spec("number", default=0.0, minimum=0.0, maximum=0.0),
                "max_sequence_length": _field_spec("integer", default=256, minimum=1, maximum=256),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
            "notes": [
                "FLUX.1-schnell is timestep-distilled: guidance_scale must be 0.",
                "max_sequence_length cannot exceed 256.",
            ],
        },
        {
            "id": "stable-diffusion-3.5-medium",
            "displayName": "Stable Diffusion 3.5 Medium",
            "modality": "image",
            "endpoint": "/generate/image/sd35",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=28, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.0, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
        },
        {
            "id": "auraflow-v0.3",
            "displayName": "AuraFlow v0.3",
            "modality": "image",
            "endpoint": "/generate/image/auraflow",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=5.0, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
        },
        {
            "id": "qwen-image-edit-2509",
            "displayName": "Qwen-Image-Edit 2509",
            "modality": "image",
            "task": "image-editing",
            "endpoint": "/generate/image/edit/qwen",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "image": _field_spec("image", required=True, description="Base64-encoded input image to edit."),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "steps": _field_spec("integer", default=20, minimum=1, maximum=60),
                "true_cfg_scale": _field_spec("number", default=4.0, minimum=0.0, maximum=20.0),
                "guidance_scale": _field_spec("number", default=1.0, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
        },
        {
            "id": "openflux-1",
            "displayName": "OpenFLUX.1",
            "modality": "image",
            "endpoint": "/generate/image/openflux",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=28, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.0, minimum=0.0, maximum=20.0),
                "max_sequence_length": _field_spec("integer", default=512, minimum=1, maximum=512),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
            "notes": [
                "This endpoint uses FluxPipeline-compatible arguments in current server implementation.",
                "Unsupported request keys are rejected via schema validation.",
            ],
        },
    ]


@router.post("/generate/image/flux")
def generate_flux(req: FluxRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            flux_runner.generate,
            prompt=req.prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            max_sequence_length=req.max_sequence_length,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _image_response(
        "flux-1-schnell",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "max_sequence_length": req.max_sequence_length,
            "num_images": req.num_images,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/image/sd35")
def generate_sd35(req: SD35Request):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            sd35_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _image_response(
        "stable-diffusion-3.5-medium",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "num_images": req.num_images,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/image/auraflow")
def generate_auraflow(req: AuraFlowRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            auraflow_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _image_response(
        "auraflow-v0.3",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "num_images": req.num_images,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/image/edit/qwen")
def generate_qwen_image_edit(req: QwenImageEditRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    input_image = _decode_input_image(req.image)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            qwen_image_edit_runner.generate,
            prompt=req.prompt,
            images=[input_image],
            output_path=str(output_path),
            negative_prompt=req.negative_prompt or " ",
            steps=req.steps,
            true_cfg_scale=req.true_cfg_scale,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _image_response(
        "qwen-image-edit-2509",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "steps": req.steps,
            "true_cfg_scale": req.true_cfg_scale,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "num_images": req.num_images,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/image/openflux")
def generate_openflux(req: OpenFluxRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            openflux_runner.generate,
            prompt=req.prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            max_sequence_length=req.max_sequence_length,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _image_response(
        "openflux-1",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "max_sequence_length": req.max_sequence_length,
            "num_images": req.num_images,
        },
        int((time.perf_counter() - start) * 1000),
    )
