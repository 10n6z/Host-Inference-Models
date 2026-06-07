from __future__ import annotations

import mimetypes
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVERS_DIR = REPO_ROOT / "servers"
COMMON_DIR = Path(__file__).resolve().parent

if str(SERVERS_DIR) not in sys.path:
    sys.path.insert(0, str(SERVERS_DIR))
if str(COMMON_DIR.parent) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR.parent))

from common.responses import build_failed_response, utc_now_iso

load_dotenv(SERVERS_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")

MODEL_CACHE_ROOT = Path(os.getenv("MODEL_CACHE_ROOT", REPO_ROOT / "model-cache")).resolve()
HF_HOME = Path(os.getenv("HF_HOME", MODEL_CACHE_ROOT / "huggingface")).resolve()
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE", HF_HOME / "hub")).resolve()
TORCH_HOME = Path(os.getenv("TORCH_HOME", MODEL_CACHE_ROOT / "torch")).resolve()
XDG_CACHE_HOME = Path(os.getenv("XDG_CACHE_HOME", MODEL_CACHE_ROOT / "cache")).resolve()
MPLCONFIGDIR = Path(os.getenv("MPLCONFIGDIR", XDG_CACHE_HOME / "matplotlib")).resolve()
HF_MODELS_ROOT = Path(os.getenv("HF_MODELS_ROOT", MODEL_CACHE_ROOT / "hf")).resolve()
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", REPO_ROOT / "outputs")).resolve()
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audio"
PUBLIC_BASE_URL = os.getenv("AUDIO_PUBLIC_BASE_URL", "http://localhost:8100").rstrip("/")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "300"))

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
TORCH_HOME.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
HF_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME))
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_HOME)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ["HF_MODELS_ROOT"] = str(HF_MODELS_ROOT)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(..., min_length=1, max_length=200)


def create_audio_family_app(
    *,
    service_name: str,
    family_name: str,
    supported_models: dict[str, dict[str, Any]],
    handlers: dict[str, Callable[..., dict[str, Any]]],
) -> FastAPI:
    app = FastAPI(title=f"{service_name} service")
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

    def _models_payload() -> list[dict[str, Any]]:
        return [
            {
                "id": model_id,
                "displayName": meta.get("display_name", model_id),
                "modality": meta.get("modality", "text-to-speech"),
                "family": family_name,
                "service": service_name,
                "fields": meta.get("fields") or {},
            }
            for model_id, meta in sorted(supported_models.items())
        ]

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(part) for part in first.get("loc", []) if part not in ("body",))
        message = first.get("msg", "Invalid request body.")
        if loc:
            message = f"{loc}: {message}"
        payload = build_failed_response(
            model="unknown",
            service=service_name,
            family=family_name,
            error_type="ValidationError",
            message=message,
            request_id=request_id,
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=422, content=payload)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": service_name,
            "family": family_name,
            "models": sorted(supported_models.keys()),
            "created_at": utc_now_iso(),
        }

    @app.get("/models")
    def models():
        return {"models": _models_payload(), "service": service_name, "family": family_name}

    @app.post("/generate")
    async def generate(request: Request):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        raw_body = await request.json()
        if not isinstance(raw_body, dict):
            payload = build_failed_response(
                model="unknown",
                service=service_name,
                family=family_name,
                error_type="ValidationError",
                message="Request body must be a JSON object.",
                request_id=request_id,
            )
            return JSONResponse(status_code=422, content=payload)

        model_id = str(raw_body.get("model", "")).strip()
        if not model_id:
            payload = build_failed_response(
                model="unknown",
                service=service_name,
                family=family_name,
                error_type="ValidationError",
                message="model: Field required",
                request_id=request_id,
            )
            return JSONResponse(status_code=422, content=payload)

        if model_id not in supported_models:
            payload = build_failed_response(
                model=model_id,
                service=service_name,
                family=family_name,
                error_type="UnknownModel",
                message=f"Model '{model_id}' is not supported by {service_name}.",
                request_id=request_id,
                details={"supported_models": sorted(supported_models.keys())},
            )
            return JSONResponse(status_code=404, content=payload)

        handler = handlers[model_id]
        params = {key: value for key, value in raw_body.items() if key != "model"}

        started = time.perf_counter()
        try:
            result = _run_with_timeout(handler, params)
        except TimeoutError as exc:
            payload = build_failed_response(
                model=model_id,
                service=service_name,
                family=family_name,
                error_type="InferenceError",
                message=str(exc),
                request_id=request_id,
            )
            return JSONResponse(status_code=504, content=payload)
        except FileNotFoundError as exc:
            payload = build_failed_response(
                model=model_id,
                service=service_name,
                family=family_name,
                error_type="ModelLoadError",
                message=str(exc),
                request_id=request_id,
            )
            return JSONResponse(status_code=503, content=payload)
        except Exception as exc:
            payload = build_failed_response(
                model=model_id,
                service=service_name,
                family=family_name,
                error_type="InferenceError",
                message=str(exc),
                request_id=request_id,
            )
            return JSONResponse(status_code=500, content=payload)

        duration_ms = int((time.perf_counter() - started) * 1000)
        result.setdefault("status", "completed")
        result.setdefault("success", True)
        result.setdefault("model_id", model_id)
        result.setdefault("modelId", model_id)
        result["duration_ms"] = duration_ms
        result["metadata"] = {
            "request_id": request_id,
            "family": family_name,
            "service": service_name,
        }
        return result

    return app


def _run_with_timeout(func: Callable[..., dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, **params)
        try:
            return future.result(timeout=INFERENCE_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            raise TimeoutError(f"Generation timed out after {int(INFERENCE_TIMEOUT_SECONDS)} seconds.") from exc


def audio_response(
    *,
    model_id: str,
    modality: str,
    audio_kind: str,
    output_id: str,
    output_path: Path,
    parameters_used: dict[str, Any],
    audio_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    relative_url = f"/outputs/audio/{output_id}"
    public_output_url = f"{PUBLIC_BASE_URL}{relative_url}"
    sample_rate = None
    audio_duration_seconds = None

    if isinstance(audio_meta, dict):
        raw_sample_rate = audio_meta.get("sample_rate")
        if raw_sample_rate is not None:
            sample_rate = int(raw_sample_rate)
        raw_duration = audio_meta.get("audio_duration_seconds", audio_meta.get("duration_seconds"))
        if raw_duration is not None:
            audio_duration_seconds = float(raw_duration)

    return {
        "success": True,
        "status": "completed",
        "modelId": model_id,
        "model_id": model_id,
        "modality": modality,
        "audio_kind": audio_kind,
        "outputType": "audio",
        "outputId": output_id,
        "outputUrl": public_output_url,
        "output_url": relative_url,
        "public_output_url": public_output_url,
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "audio/wav",
        "parameters_used": parameters_used,
        "audio_duration_seconds": audio_duration_seconds,
        "duration_seconds": audio_duration_seconds,
        "sample_rate": sample_rate,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def ensure_output_exists(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Model run completed but output file was not saved.")
