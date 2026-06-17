"""Standalone MAGI-1 video server (subprocess wrapper).

Runs in the dedicated `host-magi` conda env (torch 2.4/cu124 + flash-attn +
flashinfer). MAGI-1's supported entrypoint is a distributed CLI
(inference/pipeline/entry.py), so this server shells out to it per request with a
per-request temp config rather than importing the pipeline. Default variant is
4.5B (single GPU, Ada sm_89 supported). Final mp4 lands in the shared
OUTPUT_ROOT/videos so the gateway serves it. Supports t2v and i2v.
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import secrets
import subprocess
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MAGI_ROOT = os.getenv("MAGI_ROOT", "/gpt-lab/long/repos/MAGI-1")
MAGI_BASE_CONFIG = os.getenv("MAGI_BASE_CONFIG", f"{MAGI_ROOT}/magi_4.5b_base.json")
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "/gpt-lab/long/outputs")).resolve()
VIDEO_OUTPUT_DIR = OUTPUT_ROOT / "videos"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path(os.getenv("MAGI_TMP", "/gpt-lab/long/tmp/magi"))
TMP_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_BASE_URL = os.getenv("MAGI_PUBLIC_BASE_URL", os.getenv("PUBLIC_BASE_URL", "http://localhost:9000")).rstrip("/")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "2400"))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROMPT_MAX_LENGTH = 4000
MAX_SEED = 2_147_483_647

app = FastAPI(title="MAGI-1 Video Server")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MagiRequest(StrictRequestModel):
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_LENGTH)
    # Optional base64 image -> switches to image-to-video.
    image: Optional[str] = Field(default=None)
    num_frames: int = Field(96, ge=8, le=192)
    video_size_h: int = Field(720, ge=256, le=1024, multiple_of=8)
    video_size_w: int = Field(720, ge=256, le=1024, multiple_of=8)
    num_steps: int = Field(64, ge=4, le=128)
    cfg_number: float = Field(3.0, ge=0.0, le=20.0)
    window_size: int = Field(4, ge=1, le=16)
    fps: int = Field(24, ge=4, le=60)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    random_seed: bool = True

    @model_validator(mode="after")
    def validate_seed_requirements(self):
        if not self.random_seed and self.seed is None:
            raise ValueError("seed is required when random_seed is false.")
        return self


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: Optional[dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_seed(random_seed: bool, seed: Optional[int]) -> int:
    if random_seed:
        return secrets.randbelow(MAX_SEED + 1)
    if seed is None:
        raise APIError("VALIDATION_ERROR", "seed is required when random_seed is false.", 422)
    return int(seed)


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError):
    errors = exc.errors()
    code = "UNSUPPORTED_PARAMETER" if any(e.get("type") == "extra_forbidden" for e in errors) else "VALIDATION_ERROR"
    first = errors[0] if errors else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p not in ("body",))
    msg = first.get("msg", "Invalid request body.")
    return JSONResponse(status_code=422, content={"success": False, "code": code, "message": f"{loc}: {msg}" if loc else msg, "details": {"errors": errors}})


@app.get("/health")
def health():
    return {"status": "ok", "service": "magi-1-video-server", "created_at": _utc_now_iso()}


@app.get("/models")
def models():
    return {
        "models": [
            {
                "id": "magi-1-4.5b",
                "displayName": "MAGI-1 4.5B",
                "modality": "video",
                "endpoint": "/generate/video/magi",
                "fields": {
                    "prompt": {"type": "string", "required": True, "max_length": PROMPT_MAX_LENGTH},
                    "image": {"type": "string", "required": False},
                    "num_frames": {"type": "integer", "default": 96, "min": 8, "max": 192},
                    "video_size_h": {"type": "integer", "default": 720, "min": 256, "max": 1024, "step": 8},
                    "video_size_w": {"type": "integer", "default": 720, "min": 256, "max": 1024, "step": 8},
                    "num_steps": {"type": "integer", "default": 64, "min": 4, "max": 128},
                    "cfg_number": {"type": "number", "default": 3.0, "min": 0.0, "max": 20.0},
                    "window_size": {"type": "integer", "default": 4, "min": 1, "max": 16},
                    "fps": {"type": "integer", "default": 24, "min": 4, "max": 60},
                    "seed": {"type": "integer", "required": False, "min": 0, "max": MAX_SEED},
                    "random_seed": {"type": "boolean", "default": True},
                },
                "capabilities": {"fps": 24, "imageInput": "optional", "negativePrompt": False, "seed": True},
            }
        ]
    }


@app.post("/generate/video/magi")
def generate_magi(req: MagiRequest):
    output_id = f"vid_{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_id
    seed_used = _resolve_seed(req.random_seed, req.seed)
    job = uuid.uuid4().hex
    start = time.perf_counter()

    # Per-request config: copy the base and override runtime knobs.
    cfg = json.load(open(MAGI_BASE_CONFIG))
    rt = cfg["runtime_config"]
    rt.update(
        seed=seed_used,
        num_frames=req.num_frames,
        video_size_h=req.video_size_h,
        video_size_w=req.video_size_w,
        num_steps=req.num_steps,
        cfg_number=req.cfg_number,
        window_size=req.window_size,
        fps=req.fps,
    )
    cfg_path = TMP_DIR / f"cfg_{job}.json"
    json.dump(cfg, open(cfg_path, "w"))

    mode = "t2v"
    image_path = None
    if req.image:
        payload = req.image.split(",", 1)[-1] if req.image.startswith("data:") else req.image
        try:
            data = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise APIError("VALIDATION_ERROR", "image: invalid base64 data.", 422) from exc
        from PIL import Image

        image_path = TMP_DIR / f"img_{job}.png"
        Image.open(io.BytesIO(data)).convert("RGB").save(image_path)
        mode = "i2v"

    gpu_count = len([part for part in os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",") if part.strip()])
    if os.getenv("CUDA_VISIBLE_DEVICES", "").strip().lower() in ("", "all"):
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            gpu_count = max(1, len([line for line in smi.stdout.splitlines() if line.strip()]))
        except Exception:
            gpu_count = 1
    cmd = [
        "python3", "inference/pipeline/entry.py",
        "--config_file", str(cfg_path),
        "--mode", mode,
        "--prompt", req.prompt,
        "--output_path", str(output_path),
    ]
    if image_path:
        cmd += ["--image_path", str(image_path)]

    env = dict(os.environ)
    env.update(
        MASTER_ADDR="localhost", MASTER_PORT=os.getenv("MAGI_MASTER_PORT", "6019"),
        GPUS_PER_NODE=str(gpu_count), NNODES="1", WORLD_SIZE=str(gpu_count),
        PAD_HQ="1", PAD_DURATION="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        OFFLOAD_T5_CACHE="true", OFFLOAD_VAE_CACHE="true",
        TORCH_CUDA_ARCH_LIST="8.9;9.0",
        PYTHONPATH=f"{MAGI_ROOT}:{env.get('PYTHONPATH','')}",
    )

    if gpu_count > 1 and shutil.which("torchrun"):
        cmd = ["torchrun", f"--nproc_per_node={gpu_count}", "--master_port", env["MASTER_PORT"]] + cmd[1:]

    try:
        proc = subprocess.run(cmd, cwd=MAGI_ROOT, env=env, capture_output=True, text=True, timeout=INFERENCE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise APIError("TIMEOUT", f"Generation timed out after {int(INFERENCE_TIMEOUT_SECONDS)}s.", 504) from exc
    finally:
        cfg_path.unlink(missing_ok=True)
        if image_path:
            Path(image_path).unlink(missing_ok=True)

    if proc.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise APIError("GENERATION_FAILED", f"MAGI entry.py failed (rc={proc.returncode}): {tail}", 500)

    relative_url = f"/outputs/videos/{output_id}"
    return {
        "success": True,
        "status": "completed",
        "model_id": "magi-1-4.5b",
        "modality": "video",
        "output_url": relative_url,
        "public_output_url": f"{PUBLIC_BASE_URL}{relative_url}",
        "file_name": output_id,
        "mime_type": mimetypes.guess_type(output_path.name)[0] or "video/mp4",
        "parameters_used": {
            "prompt": req.prompt, "mode": mode, "num_frames": req.num_frames,
            "video_size_h": req.video_size_h, "video_size_w": req.video_size_w,
            "num_steps": req.num_steps, "cfg_number": req.cfg_number,
            "window_size": req.window_size, "fps": req.fps,
            "seed": seed_used, "random_seed": req.random_seed,
        },
        "duration_ms": int((time.perf_counter() - start) * 1000),
        "created_at": _utc_now_iso(),
    }
