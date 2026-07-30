"""Shared configuration, paths, and runtime helpers for the combined model server.

Importing this module loads the .env files and sets HF cache env vars, so it
must be imported before any runner module that reads those vars at import time.
"""
from __future__ import annotations

import os
import secrets
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from common import APIError

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

HF_HOME = Path(os.getenv("HF_HOME", BASE_DIR.parent / "models" / "hf-cache")).resolve()
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE", HF_HOME / "hub")).resolve()

OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", BASE_DIR.parent / "outputs")).resolve()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8001").rstrip("/")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "300"))

IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "images"
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audio"
VIDEO_OUTPUT_DIR = OUTPUT_ROOT / "videos"

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

PROMPT_MAX_LENGTH = 4000
TTS_TEXT_MAX_LENGTH = 12000
TTA_TEXT_MAX_LENGTH = 4000
IMAGE_MIN_SIZE = 512
IMAGE_MAX_SIZE = 2048
IMAGE_SIZE_STEP = 8
MAX_SEED = 2_147_483_647
MAX_NUM_IMAGES = 1
STABLE_AUDIO_MIN_DURATION_SECONDS = 1
STABLE_AUDIO_MAX_DURATION_SECONDS = 47
WAV_ONLY_PATTERN = "^(wav)$"
VIDEO_MIN_SIZE = 256
VIDEO_MAX_SIZE = 1024
VIDEO_SIZE_STEP = 8
VIDEO_MAX_FRAMES = 257

KOKORO_LANGUAGE_TO_CODE = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "pt-br": "p",
    "zh": "z",
    "zh-cn": "z",
}


def _output_url(subdir: str, file_name: str) -> str:
    return f"/outputs/{subdir}/{file_name}"


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


_CUDA_FAILURE_MARKERS = (
    "out of memory",
    "cublas_status_not_supported",
    "cublas_status_alloc_failed",
    "device-side assert",
    "cudaerrorassert",
)


def _visible_gpu_tokens() -> list[str]:
    raw = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("NVIDIA_VISIBLE_DEVICES") or ""
    if raw.strip() and raw.strip().lower() not in {"all", "none", "void"}:
        return [item.strip() for item in raw.split(",") if item.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def require_workload_gpu(env_name: str, visible_devices: list[str] | None = None) -> str:
    target_uuid = os.getenv(env_name)
    if not target_uuid:
        raise RuntimeError(f"{env_name} must be configured for a pinned workload")
    visible = _visible_gpu_tokens() if visible_devices is None else visible_devices
    if target_uuid not in visible:
        raise RuntimeError(f"{env_name}={target_uuid} is not visible to the combined server")
    return target_uuid


if os.getenv("IMAGE_GPU_UUID") or os.getenv("VIDEO_GPU_UUID"):
    require_workload_gpu("IMAGE_GPU_UUID")
    require_workload_gpu("VIDEO_GPU_UUID")


def _free_all_cached_pipelines() -> int:
    """Drop every cached GPU pipeline so the next load starts clean.

    Runners cache their pipeline on self.pipe (or self.model) and never
    evict, so cycling through models eventually exhausts VRAM. Called when a
    CUDA-level failure is detected; the failed request is retried once after.
    """
    import gc
    import sys

    freed = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith(("runners.", "routes.")) or module is None:
            continue
        for attr in vars(module).values():
            for slot in ("pipe", "model", "tts"):
                if hasattr(attr, slot) and getattr(attr, slot) is not None and hasattr(attr, "load"):
                    setattr(attr, slot, None)
                    freed += 1
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            for device_index in range(torch.cuda.device_count()):
                with torch.cuda.device(device_index):
                    torch.cuda.empty_cache()
    except Exception:
        pass
    return freed


def _is_cuda_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CUDA_FAILURE_MARKERS)


def _set_least_loaded_cuda_device() -> None:
    """Point the default 'cuda' device at the GPU with the most free memory.

    Runners load with .to('cuda'), which resolves to the current device
    (default 0). GPU 0 is often crowded by other tenants on this shared host,
    so rebind before each generation. Already-cached pipelines keep their
    explicit device and are unaffected.
    """
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return
        free_by_device = []
        for device_index in range(torch.cuda.device_count()):
            free, _total = torch.cuda.mem_get_info(device_index)
            free_by_device.append((free, device_index))
        torch.cuda.set_device(max(free_by_device)[1])
    except Exception:
        pass


def _set_workload_cuda_device(env_name: str) -> None:
    visible = _visible_gpu_tokens()
    target_uuid = require_workload_gpu(env_name, visible)
    device_index = visible.index(target_uuid)
    import torch

    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"{env_name} target {target_uuid} has no usable CUDA device")
    torch.cuda.set_device(device_index)


def _run_with_timeout(
    func,
    *args,
    timeout_seconds: float = INFERENCE_TIMEOUT_SECONDS,
    gpu_uuid_env: str | None = None,
    **kwargs,
):
    def _target():
        if gpu_uuid_env:
            _set_workload_cuda_device(gpu_uuid_env)
        else:
            _set_least_loaded_cuda_device()
        return func(*args, **kwargs)

    def _submit_once():
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_target)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError as exc:
                raise APIError(
                    code="TIMEOUT",
                    message=f"Generation timed out after {int(timeout_seconds)} seconds.",
                    status_code=504,
                ) from exc

    try:
        return _submit_once()
    except APIError:
        raise
    except Exception as exc:  # noqa: BLE001
        if not _is_cuda_failure(exc):
            raise
        freed = _free_all_cached_pipelines()
        print(f"CUDA failure detected; freed {freed} cached pipelines, retrying once. Original error: {str(exc)[:200]}")
        return _submit_once()
