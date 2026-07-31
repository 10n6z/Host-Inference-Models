"""vision-tesseract — CPU-only Tesseract OCR service (eng/fin/swe).

Contract (matches the shape validated by
sandbox/control-plane/src/services/computer-vision/modules/ocr.ts):

    POST /generate/ocr/tesseract
    Request JSON:
      { "prompt": "...", "image": "<base64>", "language": "auto"|"en"|"fi"|"sv" }
    Response JSON:
      { "words": [ { "text": "...", "confidence": 0.98,
                     "box": { "x": 10, "y": 20, "width": 100, "height": 25 } } ] }

Coordinates are in the original uploaded image's pixel space. Single-engine
output, so review_required is always False here -- the ensemble service sets
it on paddle/tesseract disagreement.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from typing import Any

import pytesseract
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("vision-tesseract")
logging.basicConfig(level=os.environ.get("VISION_TESSERACT_LOG_LEVEL", "info").upper())

MAX_IMAGE_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
MAX_IMAGE_BYTES = 12 * 1024 * 1024

# Pinned Tesseract language data, matching the plan's eng/fin/swe requirement.
DEFAULT_LANG = "eng+fin+swe"
LANGUAGE_MAP = {
    "auto": DEFAULT_LANG,
    "en": "eng",
    "fi": "fin",
    "sv": "swe",
}


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str | None = Field(default=None, max_length=4000)
    image: str = Field(..., min_length=1)
    language: str = Field(default="auto", max_length=16)
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


app = FastAPI(title="vision-tesseract", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception:  # pragma: no cover - only unreachable if the binary is missing
        version = None
    return {
        "status": "ok" if version else "degraded",
        "service": "vision-tesseract",
        "tesseract_version": version,
    }


@app.post("/generate/ocr/tesseract", response_model=OcrResponse)
def ocr(payload: OcrRequest) -> OcrResponse:
    image = _decode_image(payload.image)
    lang = LANGUAGE_MAP.get(payload.language.lower(), DEFAULT_LANG)
    try:
        raw = pytesseract.image_to_data(
            image, lang=lang, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError as exc:
        logger.exception("tesseract inference failed")
        raise HTTPException(status_code=500, detail="OCR inference failed") from exc

    words = _normalize_results(raw)
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
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="image: exceeds 25 MP")
        image.load()
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="image: exceeds pixel limit") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="image: unable to decode") from exc
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def _normalize_results(raw: dict[str, list[Any]]) -> list[OcrWord]:
    """Map ``pytesseract.image_to_data`` DICT output to the SW4E contract.

    Each index across the parallel arrays describes one detected token;
    conf is -1 for non-text regions (block/line/word boundaries) and must
    be dropped rather than coerced into a confidence value.
    """

    words: list[OcrWord] = []
    count = len(raw.get("text", []))
    for i in range(count):
        text = str(raw["text"][i]).strip()
        if not text:
            continue
        try:
            conf_raw = float(raw["conf"][i])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if conf_raw < 0:
            continue
        confidence = max(0.0, min(1.0, conf_raw / 100.0))
        try:
            box = WordBox(
                x=float(raw["left"][i]),
                y=float(raw["top"][i]),
                width=float(raw["width"][i]),
                height=float(raw["height"][i]),
            )
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if box.width <= 0 or box.height <= 0:
            continue
        words.append(OcrWord(text=text, confidence=confidence, box=box))
    return words


@app.exception_handler(HTTPException)
def _http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_type": "OcrError", "message": exc.detail},
    )
