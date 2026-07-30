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

# PP-OCRv5's multilingual latin-script bundle covers English, Finnish, Swedish.
DEFAULT_LANG = "latin"
LANGUAGE_MAP = {
    "auto": DEFAULT_LANG,
    "en": "en",
    "fi": DEFAULT_LANG,
    "sv": DEFAULT_LANG,
}

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

_ocr_pool: dict[str, Any] = {}


def _get_ocr(lang: str):
    # Lazy-import so imports do not fail readiness before the container has weights.
    from paddleocr import PaddleOCR

    if lang not in _ocr_pool:
        logger.info("loading PaddleOCR model lang=%s", lang)
        _ocr_pool[lang] = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang=lang,
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

    arr = np.array(image)  # RGB HxWx3
    try:
        raw_results = ocr_engine.predict(arr)
    except Exception as exc:  # pragma: no cover
        logger.exception("paddleocr inference failed")
        raise HTTPException(status_code=500, detail=f"OCR inference failed: {exc}") from exc

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
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="image: unable to decode") from exc
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="image: exceeds 25 MP")
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def _normalize_results(raw: Any, size: tuple[int, int]) -> list[OcrWord]:
    """Map PaddleOCR predict() output to the SW4E control-plane contract.

    PaddleX/PaddleOCR 3.x returns a list of per-page result dicts with keys
    ``rec_texts``, ``rec_scores``, and either ``rec_polys`` or ``dt_polys``.
    Polygons are already in the original image's pixel space.
    """

    width, height = size
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]

    words: list[OcrWord] = []
    for page in raw:
        page_dict = page if isinstance(page, dict) else getattr(page, "json", None) or {}
        # PaddleX result objects expose a .json attribute containing the dict
        if hasattr(page, "json") and isinstance(page.json, dict):
            page_dict = page.json.get("res") or page.json
        texts = page_dict.get("rec_texts") or []
        scores = page_dict.get("rec_scores") or []
        polys = page_dict.get("rec_polys") or page_dict.get("dt_polys") or []
        for text, score, poly in zip(texts, scores, polys):
            if not text:
                continue
            box = _poly_to_box(poly, width, height)
            if box is None:
                continue
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= confidence <= 1.0):
                continue
            words.append(
                OcrWord(text=str(text), confidence=confidence, box=WordBox(**box))
            )
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
