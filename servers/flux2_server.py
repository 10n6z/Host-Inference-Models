"""Standalone FLUX.2 [klein] image server.

Runs in the dedicated `host-models-flux2` conda env (diffusers >= 0.39) on a
separate port from the main model_server, because Flux2KleinPipeline is not
available in the shared `host-models` env. The model gateway routes
`flux-2-klein-4b` requests here. Output files are written to the shared
OUTPUT_ROOT so the gateway can serve them at /outputs/images/...
"""
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

OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", BASE_DIR.parent / "outputs")).resolve()
IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "images"
PUBLIC_BASE_URL = os.getenv("FLUX2_PUBLIC_BASE_URL", os.getenv("PUBLIC_BASE_URL", "http://localhost:8011")).rstrip("/")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "300"))

IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ConfigDict, Field, model_validator
from pydantic import BaseModel

from runners.text_to_image.flux2_klein import Flux2KleinRunner

PROMPT_MAX_LENGTH = 4000
IMAGE_MIN_SIZE = 512
IMAGE_MAX_SIZE = 2048
IMAGE_SIZE_STEP = 8
MAX_SEED = 2_147_483_647
MAX_NUM_IMAGES = 1

app = FastAPI(title="FLUX.2 klein Image Server")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

klein_runner = Flux2KleinRunner()


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: Optional[dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Flux2KleinRequest(StrictRequestModel):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    height: int = Field(1024, ge=IMAGE_MIN_SIZE, le=IMAGE_MAX_SIZE, multiple_of=IMAGE_SIZE_STEP)
    steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    random_seed: bool = True
    num_images: int = Field(1, ge=1, le=MAX_NUM_IMAGES)

    @model_validator(mode="after")
    def validate_seed_requirements(self):
        if not self.random_seed and self.seed is None:
            raise ValueError("seed is required when random_seed is false.")
        return self


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _output_url(file_name: str) -> str:
    return f"/outputs/images/{file_name}"


def _resolve_seed(random_seed: bool, seed: Optional[int]) -> int:
    if random_seed:
        return secrets.randbelow(MAX_SEED + 1)
    if seed is None:
        raise APIError("VALIDATION_ERROR", "seed is required when random_seed is false.", 422)
    return int(seed)


def _run_with_timeout(func, *args, timeout_seconds: float = INFERENCE_TIMEOUT_SECONDS, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise APIError("TIMEOUT", f"Generation timed out after {int(timeout_seconds)} seconds.", 504) from exc


def _check_output_exists(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise APIError("OUTPUT_SAVE_FAILED", "Model run completed but output file was not saved.", 500)


def _map_runtime_error(exc: Exception) -> APIError:
    if isinstance(exc, APIError):
        return exc
    lower = str(exc).lower()
    if isinstance(exc, FileNotFoundError) or "model folder not found" in lower:
        return APIError("MODEL_NOT_LOADED", str(exc), 503)
    if "out of memory" in lower:
        return APIError("CUDA_OUT_OF_MEMORY", str(exc), 507)
    return APIError("GENERATION_FAILED", str(exc), 500)


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "code": exc.code, "message": exc.message, "details": exc.details},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError):
    errors = exc.errors()
    is_unsupported = any(err.get("type") == "extra_forbidden" for err in errors)
    code = "UNSUPPORTED_PARAMETER" if is_unsupported else "VALIDATION_ERROR"
    first = errors[0] if errors else {}
    loc = ".".join(str(part) for part in first.get("loc", []) if part not in ("body",))
    message = first.get("msg", "Invalid request body.")
    if loc:
        message = f"{loc}: {message}"
    return JSONResponse(
        status_code=422,
        content={"success": False, "code": code, "message": message, "details": {"errors": errors}},
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "flux2-klein-image-server", "created_at": _utc_now_iso()}


@app.get("/models")
def models():
    return {
        "models": [
            {
                "id": "flux-2-klein-4b",
                "displayName": "FLUX.2 klein 4B",
                "modality": "image",
                "endpoint": "/generate/image/flux2-klein",
                "fields": {
                    "prompt": {"type": "string", "required": True, "max_length": PROMPT_MAX_LENGTH},
                    "width": {"type": "integer", "default": 1024, "min": IMAGE_MIN_SIZE, "max": IMAGE_MAX_SIZE, "step": IMAGE_SIZE_STEP},
                    "height": {"type": "integer", "default": 1024, "min": IMAGE_MIN_SIZE, "max": IMAGE_MAX_SIZE, "step": IMAGE_SIZE_STEP},
                    "steps": {"type": "integer", "default": 4, "min": 1, "max": 50},
                    "guidance_scale": {"type": "number", "default": 1.0, "min": 0.0, "max": 20.0},
                    "seed": {"type": "integer", "required": False, "min": 0, "max": MAX_SEED},
                    "random_seed": {"type": "boolean", "default": True},
                },
            }
        ]
    }


@app.post("/generate/image/flux2-klein")
def generate_flux2_klein(req: Flux2KleinRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    start = time.perf_counter()

    try:
        _run_with_timeout(
            klein_runner.generate,
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

    relative_url = _output_url(output_id)
    output_url = f"{PUBLIC_BASE_URL}{relative_url}"
    return {
        "success": True,
        "status": "completed",
        "modelId": "flux-2-klein-4b",
        "model_id": "flux-2-klein-4b",
        "modality": "image",
        "outputType": "image",
        "outputId": output_id,
        "outputUrl": output_url,
        "output_url": relative_url,
        "public_output_url": output_url,
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "image/png",
        "parameters_used": {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed_used,
            "random_seed": req.random_seed,
            "num_images": req.num_images,
        },
        "duration_ms": int((time.perf_counter() - start) * 1000),
        "created_at": _utc_now_iso(),
    }
