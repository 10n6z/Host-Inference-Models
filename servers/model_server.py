"""Combined GPU inference server (image + video + audio) on a single app.

Per-modality request schemas, runners, registries, and endpoints live in
servers/routes/{image,video,audio}.py. This module wires them onto one FastAPI
app, exposes /health and an aggregated /models, and owns the error handlers.

`config` is imported first so HF cache env vars are set before the route
modules import their runners.
"""
import config  # noqa: F401  (import for env side effects; must precede runner imports)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from common import APIError, _utc_now_iso, _validation_message
from routes import audio, image, video

# Re-exported so tests can monkeypatch runner instances via the model_server module.
flux_runner = image.flux_runner
sd35_runner = image.sd35_runner
auraflow_runner = image.auraflow_runner
openflux_runner = image.openflux_runner
kokoro_runner = audio.kokoro_runner
stable_audio_open_runner = audio.stable_audio_open_runner
wan_t2v_runner = video.wan_t2v_runner
cogvideox_runner = video.cogvideox_runner
ltx_video_runner = video.ltx_video_runner

app = FastAPI(title="Local AI GPU Inference Server")
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_ROOT)), name="outputs")

app.include_router(image.router)
app.include_router(video.router)
app.include_router(audio.router)


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
        "service": "local-ai-gpu-inference-server",
        "created_at": _utc_now_iso(),
    }


@app.get("/models")
def models():
    return {"models": image.model_registry() + video.model_registry() + audio.model_registry()}
