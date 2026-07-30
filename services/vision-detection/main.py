"""vision-detection — RT-DETR r50vd object-detection service.

Contract (matches
sandbox/control-plane/src/services/computer-vision/modules/detection.ts):

    POST /generate/detect/rtdetr
    Request JSON:
      { "prompt": "...", "image": "<base64>", "confidence_threshold": 0.5 }
    Response JSON:
      { "detections": [ { "label": "person", "confidence": 0.94,
                          "box": { "x": 20, "y": 30, "width": 150, "height": 300 } } ] }

Coordinates are in the original uploaded image's pixel space, produced by
``AutoImageProcessor.post_process_object_detection(target_sizes=[(H, W)])``.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("vision-detection")
logging.basicConfig(level=os.environ.get("VISION_DETECTION_LOG_LEVEL", "info").upper())

MODEL_ID = os.environ.get("VISION_DETECTION_MODEL_ID", "PekingU/rtdetr_r50vd")
DEFAULT_THRESHOLD = 0.5
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

_model_state: dict[str, Any] = {}


def _load_model() -> None:
    if _model_state:
        return
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    logger.info("loading detection model %s", MODEL_ID)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _model_state["processor"] = processor
    _model_state["model"] = model
    _model_state["device"] = device
    _model_state["torch"] = torch


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str | None = Field(default=None, max_length=4000)
    image: str = Field(..., min_length=1)
    confidence_threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    task: str | None = None
    model: str | None = None


class DetectionBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Detection(BaseModel):
    label: str
    confidence: float
    box: DetectionBox


class DetectionResponse(BaseModel):
    detections: list[Detection]


app = FastAPI(title="vision-detection", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "vision-detection",
        "model_id": MODEL_ID,
        "loaded": bool(_model_state),
        "device": _model_state.get("device"),
    }


@app.post("/generate/detect/rtdetr", response_model=DetectionResponse)
def detect(payload: DetectRequest) -> DetectionResponse:
    image = _decode_image(payload.image)
    _load_model()
    torch = _model_state["torch"]
    processor = _model_state["processor"]
    model = _model_state["model"]
    device = _model_state["device"]

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    height, width = image.height, image.width
    target_sizes = torch.tensor([(height, width)], device=device)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=float(payload.confidence_threshold),
    )
    if not results:
        return DetectionResponse(detections=[])
    result = results[0]

    id2label = getattr(model.config, "id2label", {}) or {}
    detections: list[Detection] = []
    scores = result.get("scores")
    labels = result.get("labels")
    boxes = result.get("boxes")
    if scores is None or labels is None or boxes is None:
        return DetectionResponse(detections=[])

    for score, label_idx, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        label_name = id2label.get(int(label_idx), str(int(label_idx)))
        if not label_name:
            continue
        confidence = float(score)
        if not (0.0 <= confidence <= 1.0):
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        x = max(0.0, min(x0, width))
        y = max(0.0, min(y0, height))
        w = max(0.0, min(x1, width) - x)
        h = max(0.0, min(y1, height) - y)
        if w <= 0 or h <= 0:
            continue
        detections.append(
            Detection(
                label=str(label_name),
                confidence=confidence,
                box=DetectionBox(x=x, y=y, width=w, height=h),
            )
        )
    return DetectionResponse(detections=detections)


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


@app.exception_handler(HTTPException)
def _http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_type": "DetectionError", "message": exc.detail},
    )
