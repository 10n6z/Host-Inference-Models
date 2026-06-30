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
from runners.text_to_image.dreamshaper_xl import DreamShaperXLRunner
from runners.text_to_image.dreamshaper8 import DreamShaper8Runner
from runners.text_to_image.flux_schnell import FluxSchnellRunner
from runners.text_to_image.kandinsky22 import Kandinsky22Runner
from runners.text_to_image.lcm_sdxl import LCMSDXLRunner
from runners.text_to_image.openflux import OpenFluxRunner
from runners.text_to_image.openjourney_v4 import OpenjourneyV4Runner
from runners.text_to_image.pixart_sigma import PixArtSigmaXL2Runner
from runners.text_to_image.playground_v25 import PlaygroundV25Runner
from runners.text_to_image.realvisxl import RealVisXLRunner
from runners.text_to_image.sd35_large_turbo import SD35LargeTurboRunner
from runners.text_to_image.sd35_medium import SD35MediumRunner
from runners.text_to_image.sdxl_lightning import SDXLLightningRunner
from runners.text_to_image.sd15 import SD15Runner
from runners.text_to_image.segmind_vega import SegmindVegaRunner
from runners.text_to_image.ssd1b import SSD1BRunner
from runners.text_to_image.wuerstchen_v2 import WuerstchenV2Runner
from runners.text_to_image.sd21 import SD21Runner
from runners.text_to_image.sdxl_base import SDXLBaseRunner
from runners.image_to_image.qwen_image_edit import QwenImageEditRunner
from runners.image_to_image.sd15_inpaint import SD15InpaintRunner

router = APIRouter()

flux_runner = FluxSchnellRunner()
sd35_runner = SD35MediumRunner()
auraflow_runner = AuraFlowRunner()
openflux_runner = OpenFluxRunner()
qwen_image_edit_runner = QwenImageEditRunner()
sd15_inpaint_runner = SD15InpaintRunner()
sd15_runner = SD15Runner()
sd21_runner = SD21Runner()
sdxl_base_runner = SDXLBaseRunner()
dreamshaper8_runner = DreamShaper8Runner()
sdxl_lightning_runner = SDXLLightningRunner()
dreamshaper_xl_runner = DreamShaperXLRunner()
sd35_large_turbo_runner = SD35LargeTurboRunner()
realvisxl_runner = RealVisXLRunner()
playground_v25_runner = PlaygroundV25Runner()
ssd1b_runner = SSD1BRunner()
segmind_vega_runner = SegmindVegaRunner()
lcm_sdxl_runner = LCMSDXLRunner()
openjourney_v4_runner = OpenjourneyV4Runner()
pixart_sigma_runner = PixArtSigmaXL2Runner()
kandinsky22_runner = Kandinsky22Runner()
wuerstchen_v2_runner = WuerstchenV2Runner()


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


class SD15InpaintRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    image: str = Field(..., min_length=1)
    mask_image: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(512, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(512, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    strength: float = Field(1.0, ge=0.0, le=1.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)

class SD15Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(512, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(512, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class SD21Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(768, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(768, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class DreamShaper8Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(512, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(512, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class SDXLBaseRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(7.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class SDXLLightningRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(4, ge=1, le=8)
    guidance_scale: float = Field(0.0, ge=0.0, le=0.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class DreamShaperXLRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(7.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class RealVisXLRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(7.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class PlaygroundV25Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(3.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class SSD1BRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(9.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class SegmindVegaRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(25, ge=1, le=60)
    guidance_scale: float = Field(9.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class LCMSDXLRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(4, ge=1, le=12)
    guidance_scale: float = Field(1.0, ge=0.0, le=10.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class OpenjourneyV4Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(512, ge=IMAGE_MIN_SIZE, le=1024, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(512, ge=IMAGE_MIN_SIZE, le=1024, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(7.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class PixArtSigmaRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(20, ge=1, le=60)
    guidance_scale: float = Field(4.5, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class Kandinsky22Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(50, ge=1, le=100)
    guidance_scale: float = Field(4.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class WuerstchenV2Request(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=1536, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(4.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)


class SD35LargeTurboRequest(ImageSeedMixin):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: Optional[str] = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    max_sequence_length: int = Field(512, ge=1, le=512)
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
            "id": "sd-1-5-inpainting",
            "displayName": "Stable Diffusion 1.5 Inpainting",
            "modality": "image",
            "task": "image-inpainting",
            "endpoint": "/generate/image/inpaint/sd15",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "image": _field_spec("image", required=True, description="Base64-encoded base image to inpaint."),
                "mask_image": _field_spec("image", required=True, description="Base64-encoded mask (white = regenerate, black = keep)."),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=512, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=512, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.5, minimum=0.0, maximum=20.0),
                "strength": _field_spec("number", default=1.0, minimum=0.0, maximum=1.0),
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
        {
            "id": "sd-v1-5",
            "displayName": "Stable Diffusion v1.5",
            "modality": "image",
            "endpoint": "/generate/image/sd15",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=512, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=512, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.5, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
        },
        {
            "id": "dreamshaper-8",
            "displayName": "DreamShaper 8",
            "modality": "image",
            "endpoint": "/generate/image/dreamshaper-8",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=512, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=512, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.5, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
        },
        {
            "id": "sd-2-1",
            "displayName": "Stable Diffusion 2.1",
            "modality": "image",
            "endpoint": "/generate/image/sd21",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=768, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=768, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.5, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
        },
        {
            "id": "sdxl-base-1.0",
            "displayName": "Stable Diffusion XL Base 1.0",
            "modality": "image",
            "endpoint": "/generate/image/sdxl",
            "fields": {
                "prompt": _field_spec("string", required=True, max_length=PROMPT_MAX_LENGTH),
                "negative_prompt": _field_spec("string", required=False, max_length=PROMPT_MAX_LENGTH),
                "width": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "height": _field_spec("integer", default=1024, minimum=IMAGE_MIN_SIZE, maximum=IMAGE_MAX_SIZE, step=IMAGE_SIZE_STEP),
                "steps": _field_spec("integer", default=30, minimum=1, maximum=60),
                "guidance_scale": _field_spec("number", default=7.0, minimum=0.0, maximum=20.0),
                "seed": _field_spec("integer", required=False, minimum=0, maximum=MAX_SEED),
                "random_seed": _field_spec("boolean", default=True),
                "num_images": _field_spec("integer", default=1, minimum=1, maximum=MAX_NUM_IMAGES),
            },
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


@router.post("/generate/image/inpaint/sd15")
def generate_sd15_inpaint(req: SD15InpaintRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    base_image = _decode_input_image(req.image)
    mask_image = _decode_input_image(req.mask_image)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            sd15_inpaint_runner.generate,
            prompt=req.prompt,
            image=base_image,
            mask_image=mask_image,
            output_path=str(output_path),
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            strength=req.strength,
            seed=seed_used,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _image_response(
        "sd-1-5-inpainting",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "strength": req.strength,
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


def _sd_like_generate(runner, model_id: str, req) -> dict[str, Any]:
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
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
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed_used,
            num_images=req.num_images,
        )
        _check_output_exists(output_path)
    except Exception as exc:
        raise _map_runtime_error(exc)

    return _image_response(
        model_id,
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


@router.post("/generate/image/sd15")
def generate_sd15(req: SD15Request):
    return _sd_like_generate(sd15_runner, "sd-v1-5", req)


@router.post("/generate/image/sd21")
def generate_sd21(req: SD21Request):
    return _sd_like_generate(sd21_runner, "sd-2-1", req)


@router.post("/generate/image/dreamshaper-8")
def generate_dreamshaper8(req: DreamShaper8Request):
    return _sd_like_generate(dreamshaper8_runner, "dreamshaper-8", req)


@router.post("/generate/image/sdxl")
def generate_sdxl_base(req: SDXLBaseRequest):
    return _sd_like_generate(sdxl_base_runner, "sdxl-base-1.0", req)


@router.post("/generate/image/sdxl-lightning")
def generate_sdxl_lightning(req: SDXLLightningRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            sdxl_lightning_runner.generate,
            prompt=req.prompt,
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
        "sdxl-lightning",
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
            "num_images": req.num_images,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/image/dreamshaper-xl")
def generate_dreamshaper_xl(req: DreamShaperXLRequest):
    return _sd_like_generate(dreamshaper_xl_runner, "dreamshaper-xl", req)


@router.post("/generate/image/realvisxl-v4")
def generate_realvisxl(req: RealVisXLRequest):
    return _sd_like_generate(realvisxl_runner, "realvisxl-v4", req)


@router.post("/generate/image/playground-v2-5")
def generate_playground_v25(req: PlaygroundV25Request):
    return _sd_like_generate(playground_v25_runner, "playground-v2-5-1024", req)


@router.post("/generate/image/ssd-1b")
def generate_ssd1b(req: SSD1BRequest):
    return _sd_like_generate(ssd1b_runner, "ssd-1b", req)


@router.post("/generate/image/segmind-vega")
def generate_segmind_vega(req: SegmindVegaRequest):
    return _sd_like_generate(segmind_vega_runner, "segmind-vega", req)


@router.post("/generate/image/openjourney")
def generate_openjourney_v4(req: OpenjourneyV4Request):
    return _sd_like_generate(openjourney_v4_runner, "openjourney-v4", req)


@router.post("/generate/image/pixart-sigma")
def generate_pixart_sigma(req: PixArtSigmaRequest):
    return _sd_like_generate(pixart_sigma_runner, "pixart-sigma-xl-2", req)


@router.post("/generate/image/lcm-sdxl")
def generate_lcm_sdxl(req: LCMSDXLRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            lcm_sdxl_runner.generate,
            prompt=req.prompt,
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
        "lcm-sdxl",
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
            "num_images": req.num_images,
        },
        int((time.perf_counter() - start) * 1000),
    )


@router.post("/generate/image/kandinsky-2-2")
def generate_kandinsky22(req: Kandinsky22Request):
    return _sd_like_generate(kandinsky22_runner, "kandinsky-2-2-decoder", req)


@router.post("/generate/image/wuerstchen")
def generate_wuerstchen_v2(req: WuerstchenV2Request):
    return _sd_like_generate(wuerstchen_v2_runner, "wuerstchen-v2", req)


@router.post("/generate/image/sd35-large-turbo")
def generate_sd35_large_turbo(req: SD35LargeTurboRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            sd35_large_turbo_runner.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
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
        "sd-3.5-large-turbo",
        output_id,
        output_path,
        {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "max_sequence_length": req.max_sequence_length,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "num_images": req.num_images,
        },
        int((time.perf_counter() - start) * 1000),
    )
