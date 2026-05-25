from __future__ import annotations

import mimetypes
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

HF_HOME = Path(os.getenv("HF_HOME", BASE_DIR.parent / "models" / "hf-cache")).resolve()
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE", HF_HOME / "hub")).resolve()
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", BASE_DIR.parent / "outputs")).resolve()
IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "images"
PUBLIC_BASE_URL = os.getenv("IMAGE_PUBLIC_BASE_URL", os.getenv("PUBLIC_BASE_URL", "http://localhost:8001")).rstrip("/")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "300"))

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from runners.text_to_image.auraflow import AuraFlowRunner
from runners.text_to_image.flux_schnell import FluxSchnellRunner
from runners.text_to_image.openflux import OpenFluxRunner
from runners.text_to_image.sd35_medium import SD35MediumRunner

PROMPT_MAX_LENGTH = 4000
IMAGE_MIN_SIZE = 512
IMAGE_MAX_SIZE = 2048
IMAGE_SIZE_STEP = 8
MAX_SEED = 2_147_483_647
MAX_NUM_IMAGES = 1

app = FastAPI(title="Local AI Image Generation Server")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

flux_runner = FluxSchnellRunner()
sd35_runner = SD35MediumRunner()
auraflow_runner = AuraFlowRunner()
openflux_runner = OpenFluxRunner()


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: Optional[dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


def _field_spec(
    field_type: str,
    *,
    required: Optional[bool] = None,
    default: Any = None,
    minimum: Any = None,
    maximum: Any = None,
    step: Any = None,
    enum: Optional[list[Any]] = None,
    max_length: Optional[int] = None,
    description: Optional[str] = None,
):
    data: dict[str, Any] = {"type": field_type}
    if required is not None:
        data["required"] = required
    if default is not None:
        data["default"] = default
    if minimum is not None:
        data["min"] = minimum
    if maximum is not None:
        data["max"] = maximum
    if step is not None:
        data["step"] = step
    if enum is not None:
        data["enum"] = enum
    if max_length is not None:
        data["max_length"] = max_length
    if description:
        data["description"] = description
    return data


def _model_registry() -> list[dict[str, Any]]:
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
        },
    ]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _output_url(file_name: str) -> str:
    return f"/outputs/images/{file_name}"


def _public_output_url(relative_output_url: str) -> str:
    return f"{PUBLIC_BASE_URL}{relative_output_url}"


def _resolve_seed(random_seed: bool, seed: Optional[int]) -> int:
    if random_seed:
        return secrets.randbelow(MAX_SEED + 1)
    if seed is None:
        raise APIError(
            code="VALIDATION_ERROR",
            message="seed is required when random_seed is false.",
            status_code=422,
            details={"seed": "Provide seed or set random_seed=true."},
        )
    return int(seed)


def _run_with_timeout(func, *args, timeout_seconds: float = INFERENCE_TIMEOUT_SECONDS, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise APIError(
                code="TIMEOUT",
                message=f"Generation timed out after {int(timeout_seconds)} seconds.",
                status_code=504,
            ) from exc


def _check_output_exists(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise APIError(
            code="OUTPUT_SAVE_FAILED",
            message="Model run completed but output file was not saved.",
            status_code=500,
            details={"output_path": str(path)},
        )


def _map_runtime_error(exc: Exception) -> APIError:
    message = str(exc)
    lower = message.lower()

    if isinstance(exc, APIError):
        return exc
    if isinstance(exc, FileNotFoundError) or "model folder not found" in lower:
        return APIError("MODEL_NOT_LOADED", message, 503)
    if "cuda out of memory" in lower or "out of memory" in lower:
        return APIError("CUDA_OUT_OF_MEMORY", message, 507)
    return APIError("GENERATION_FAILED", message, 500)


def _validation_message(errors: list[dict[str, Any]]) -> str:
    first = errors[0] if errors else {}
    loc = first.get("loc", [])
    loc_text = ".".join(str(part) for part in loc if part not in ("body",))
    msg = first.get("msg", "Invalid request body.")
    if loc_text:
        return f"{loc_text}: {msg}"
    return msg


def _image_response(
    *,
    model_id: str,
    output_id: str,
    output_path: Path,
    parameters_used: dict[str, Any],
    duration_ms: int,
) -> dict[str, Any]:
    relative_url = _output_url(output_id)
    output_url = _public_output_url(relative_url)
    return {
        "success": True,
        "status": "completed",
        "modelId": model_id,
        "model_id": model_id,
        "modality": "image",
        "outputType": "image",
        "outputId": output_id,
        "outputUrl": output_url,
        "output_url": relative_url,
        "public_output_url": output_url,
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "image/png",
        "parameters_used": parameters_used,
        "duration_ms": duration_ms,
        "created_at": _utc_now_iso(),
    }


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError):
    errors = exc.errors()
    is_unsupported = any(err.get("type") == "extra_forbidden" for err in errors)
    code = "UNSUPPORTED_PARAMETER" if is_unsupported else "VALIDATION_ERROR"
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "code": code,
            "message": _validation_message(errors),
            "details": {"errors": errors},
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "local-ai-image-generation-server",
        "created_at": _utc_now_iso(),
    }


@app.get("/models")
def models():
    return {"models": _model_registry()}


@app.post("/generate/image/flux")
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
        model_id="flux-1-schnell",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
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
        duration_ms=int((time.perf_counter() - start) * 1000),
    )


@app.post("/generate/image/sd35")
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
        model_id="stable-diffusion-3.5-medium",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
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
        duration_ms=int((time.perf_counter() - start) * 1000),
    )


@app.post("/generate/image/auraflow")
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
        model_id="auraflow-v0.3",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
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
        duration_ms=int((time.perf_counter() - start) * 1000),
    )


@app.post("/generate/image/openflux")
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
        model_id="openflux-1",
        output_id=output_id,
        output_path=output_path,
        parameters_used={
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
        duration_ms=int((time.perf_counter() - start) * 1000),
    )
