"""Shared configuration, paths, and runtime helpers for the combined model server.

Importing this module loads the .env files and sets HF cache env vars, so it
must be imported before any runner module that reads those vars at import time.
"""
from __future__ import annotations

import os
import secrets
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
