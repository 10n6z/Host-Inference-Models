"""vision-ocr-ensemble — combines PaddleOCR and Tesseract results.

Contract:

    POST /generate/ocr/ensemble
    Request JSON:
      { "prompt": "...", "image": "<base64>", "language": "auto"|"en"|"fi"|"sv" }
    Response JSON:
      { "words": [ { "text": "...", "confidence": 0.91, "box": {...},
                     "review_required": false } ] }

Calls vision-ocr and vision-tesseract concurrently, spatially aligns their
words by IoU, picks the higher-confidence reading for each matched pair, and
flags a pair for human review when the two engines disagree on the text.
Unmatched words from either engine pass through unreviewed -- there is
nothing to disagree with when only one engine saw a token.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("vision-ocr-ensemble")
logging.basicConfig(level=os.environ.get("VISION_OCR_ENSEMBLE_LOG_LEVEL", "info").upper())

PADDLE_URL = os.environ.get(
    "VISION_OCR_PADDLE_URL", "http://vision-ocr:8120/generate/ocr/paddle"
)
TESSERACT_URL = os.environ.get(
    "VISION_OCR_TESSERACT_URL", "http://vision-tesseract:8122/generate/ocr/tesseract"
)
UPSTREAM_TIMEOUT_SECONDS = float(os.environ.get("VISION_OCR_ENSEMBLE_TIMEOUT", "30"))

# Below this spatial overlap, two boxes are treated as different words rather
# than two engines' readings of the same token.
IOU_MATCH_THRESHOLD = 0.3


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
    review_required: bool = False


class OcrResponse(BaseModel):
    words: list[OcrWord]


app = FastAPI(title="vision-ocr-ensemble", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "vision-ocr-ensemble"}


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _box_iou(a: WordBox, b: WordBox) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.width, a.y + a.height
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.width, b.y + b.height
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def choose_word(paddle: OcrWord, tesseract: OcrWord) -> OcrWord:
    selected = paddle if paddle.confidence >= tesseract.confidence else tesseract
    return selected.model_copy(
        update={
            "review_required": _normalize(paddle.text) != _normalize(tesseract.text),
        }
    )


def combine_words(paddle: list[OcrWord], tesseract: list[OcrWord]) -> dict[str, Any]:
    matched_tesseract_indices: set[int] = set()
    combined: list[OcrWord] = []

    for paddle_word in paddle:
        best_index = -1
        best_iou = IOU_MATCH_THRESHOLD
        for index, tesseract_word in enumerate(tesseract):
            if index in matched_tesseract_indices:
                continue
            iou = _box_iou(paddle_word.box, tesseract_word.box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0:
            matched_tesseract_indices.add(best_index)
            combined.append(choose_word(paddle_word, tesseract[best_index]))
        else:
            combined.append(paddle_word)

    for index, tesseract_word in enumerate(tesseract):
        if index not in matched_tesseract_indices:
            combined.append(tesseract_word)

    return {"words": [word.model_dump() for word in combined]}


async def _call_engine(client: httpx.AsyncClient, url: str, payload: dict[str, Any]) -> list[OcrWord]:
    try:
        response = await client.post(url, json=payload, timeout=UPSTREAM_TIMEOUT_SECONDS)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("ensemble upstream call failed url=%s error=%s", url, exc)
        return []
    data = response.json()
    return [OcrWord(**word) for word in data.get("words", [])]


@app.post("/generate/ocr/ensemble", response_model=OcrResponse)
async def ocr(payload: OcrRequest) -> OcrResponse:
    body = {"prompt": payload.prompt, "image": payload.image, "language": payload.language}
    async with httpx.AsyncClient() as client:
        paddle_words = await _call_engine(client, PADDLE_URL, body)
        tesseract_words = await _call_engine(client, TESSERACT_URL, body)

    if not paddle_words and not tesseract_words:
        raise HTTPException(status_code=502, detail="Both OCR engines were unavailable")

    combined = combine_words(paddle_words, tesseract_words)
    return OcrResponse(**combined)


@app.exception_handler(HTTPException)
def _http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_type": "OcrError", "message": exc.detail},
    )
