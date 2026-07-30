"""vision-ocr — PaddleOCR PP-OCRv5 multilingual OCR service.

Contract (matches the shape validated by
sandbox/control-plane/src/services/computer-vision/modules/ocr.ts):

    POST /generate/ocr/paddle
    Request JSON:
      { "prompt": "...", "image": "<base64>", "language": "auto"|"en"|"fi"|"sv" }
    Response JSON:
      { "words": [ { "text": "...", "confidence": 0.98,
                     "box": { "x": 10, "y": 20, "width": 100, "height": 25 } } ] }

Coordinates are in the original uploaded image's pixel space.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("vision-ocr")
logging.basicConfig(level=os.environ.get("VISION_OCR_LOG_LEVEL", "info").upper())

MAX_IMAGE_PIXELS = 25_000_000
# Align Pillow's own bomb guard with our limit so a crafted image cannot decode
# a buffer larger than we would accept.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# PP-OCRv5's multilingual latin-script bundle covers English, Finnish, Swedish.
DEFAULT_LANG = "latin"
LANGUAGE_MAP = {
    "auto": DEFAULT_LANG,
    "en": "en",
    "fi": DEFAULT_LANG,
    "sv": DEFAULT_LANG,
}

MAX_IMAGE_BYTES = 12 * 1024 * 1024

_ocr_pool: dict[str, Any] = {}


def _get_ocr(lang: str):
    # Lazy-import so imports do not fail readiness before the container has weights.
    from paddleocr import PaddleOCR

    if lang not in _ocr_pool:
        logger.info("loading PaddleOCR model lang=%s", lang)
        _ocr_pool[lang] = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
        )
    return _ocr_pool[lang]


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str | None = Field(default=None, max_length=4000)
    image: str = Field(..., min_length=1)
    language: str = Field(default="auto", max_length=16)
    # tolerated for gateway payload passthrough
    task: str | None = None
    model: str | None = None


class WordBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class OcrWord(BaseModel):
    text: str
    confidence: float
    box: WordBox


class OcrResponse(BaseModel):
    words: list[OcrWord]


app = FastAPI(title="vision-ocr", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "vision-ocr",
        "loaded_langs": sorted(_ocr_pool.keys()),
    }


@app.post("/generate/ocr/paddle", response_model=OcrResponse)
def ocr(payload: OcrRequest) -> OcrResponse:
    image = _decode_image(payload.image)
    lang = LANGUAGE_MAP.get(payload.language.lower(), DEFAULT_LANG)
    ocr_engine = _get_ocr(lang)

    # PaddleOCR expects BGR channel order when handed a numpy array (OpenCV
    # convention). PIL decodes RGB, so swap channels or R/B are inverted.
    arr = np.array(image)[:, :, ::-1]  # RGB -> BGR, HxWx3
    try:
        raw_results = ocr_engine.ocr(arr, cls=True)
    except Exception as exc:  # pragma: no cover
        logger.exception("paddleocr inference failed")
        raise HTTPException(status_code=500, detail="OCR inference failed") from exc

    words = _normalize_results(raw_results, image.size)
    return OcrResponse(words=words)


def _decode_image(b64: str) -> Image.Image:
    try:
        payload = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image: not valid base64") from exc
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image: exceeds 12 MB")
    try:
        buffer = io.BytesIO(payload)
        image = Image.open(buffer)
        # Reject by declared dimensions BEFORE load() allocates the pixel buffer,
        # so a small file declaring huge dimensions cannot OOM the container.
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="image: exceeds 25 MP")
        image.load()
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="image: exceeds pixel limit") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="image: unable to decode") from exc
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def _normalize_results(raw: Any, size: tuple[int, int]) -> list[OcrWord]:
    """Map PaddleOCR 2.7.x ``.ocr()`` output to the SW4E control-plane contract.

    Output shape is a list per input image; each image is a list of
    ``[box_poly, (text, confidence)]`` entries. Polygons are already in the
    original image's pixel space.
    """

    width, height = size
    if not raw:
        return []

    words: list[OcrWord] = []
    dropped = 0
    seen = 0
    for page in raw:
        if not page:
            continue
        for entry in page:
            seen += 1
            if not entry or len(entry) < 2:
                dropped += 1
                continue
            poly, recognized = entry[0], entry[1]
            if not recognized or len(recognized) < 2:
                dropped += 1
                continue
            text, score = recognized[0], recognized[1]
            if not text:
                dropped += 1
                continue
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (0.0 <= confidence <= 1.0):
                dropped += 1
                continue
            box = _poly_to_box(poly, width, height)
            if box is None:
                dropped += 1
                continue
            words.append(
                OcrWord(text=str(text), confidence=confidence, box=WordBox(**box))
            )
    if dropped:
        # A high drop rate usually means the PaddleOCR output shape changed
        # (e.g. a 2.7.x -> 3.x upgrade), which would otherwise look like
        # "no text found" to the caller.
        logger.warning("ocr dropped %d of %d entries during normalization", dropped, seen)
    return words


def _poly_to_box(poly: Any, width: int, height: int) -> dict[str, float] | None:
    try:
        pts = np.asarray(poly, dtype=float)
    except (TypeError, ValueError):
        return None
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 3:
        return None
    xs, ys = pts[:, 0], pts[:, 1]
    x0 = float(np.clip(xs.min(), 0, width))
    y0 = float(np.clip(ys.min(), 0, height))
    x1 = float(np.clip(xs.max(), 0, width))
    y1 = float(np.clip(ys.max(), 0, height))
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    return {"x": x0, "y": y0, "width": w, "height": h}


@app.exception_handler(HTTPException)
def _http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_type": "OcrError", "message": exc.detail},
    )
