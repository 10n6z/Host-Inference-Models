"""vision-grounding-dino — OWLv2 open-vocabulary detection service.

Contract (matches sandbox's detectorRouting.ts "grounding-dino" model id, routed
for jobs with requestedLabels) is unchanged, but the model backing it is not
Grounding DINO anymore -- see "Model swap" below.

    POST /generate/detect/grounding-dino
    Request JSON:
      { "prompt": "...", "image": "<base64>",
        "labels": ["forklift", "ladder"],       # optional; open-vocabulary queries
        "confidence_threshold": 0.4,             # detection score threshold
        "text_threshold": 0.3 }                  # accepted for compat, unused
    Response JSON:
      { "detections": [ { "label": "forklift", "confidence": 0.81,
                          "box": { "x": 20, "y": 30, "width": 150, "height": 300 } } ] }

If "labels" is omitted, "prompt" is split on periods/commas into candidate phrases
(OWLv2 takes a flat list of label strings, not a Grounding-DINO-style
period-terminated phrase string).

Coordinates are in the original uploaded image's pixel space, produced by
``Owlv2Processor.post_process_object_detection(target_sizes=[(H, W)])``.

Model swap (2026-08-02): replaced Grounding DINO tiny with OWLv2
(google/owlv2-base-patch16-ensemble). Grounding-DINO-tiny's recall50 on the
Phase 2 LVIS open-vocabulary eval plateaued at ~0.43-0.50 no matter how the
box/text thresholds were tuned (measured on the full 451-box corpus, not a
sample) -- a real capacity limit, not a confidence-threshold problem.
grounding-dino-base was tried too and scored *worse* (~0.31-0.43). OWLv2-base
measured 0.7162 recall50 at threshold=0.05 on the same full corpus, clearing
the 0.60 quality floor with real margin. `text_threshold` is kept in the
request/response contract for backward compatibility but has no effect on
OWLv2, which only has one score threshold.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from vision_common_metrics import VISION_METRICS_CONTENT_TYPE, build_vision_metrics

logger = logging.getLogger("vision-grounding-dino")
logging.basicConfig(level=os.environ.get("VISION_GROUNDING_DINO_LOG_LEVEL", "info").upper())

MODEL_ID = os.environ.get("VISION_GROUNDING_DINO_MODEL_ID", "google/owlv2-base-patch16-ensemble")
# Measured on the full Phase 2 open-vocabulary corpus (451 boxes): 0.05 gives
# 0.7162 recall50 vs the 0.60 floor; 0.10 gives 0.5765 (misses); 0.15 gives
# 0.4745 (misses further). Lower is not free -- it trades precision for
# recall -- but this floor only scores recall, and 0.05 leaves real margin
# rather than sitting right at the line.
DEFAULT_BOX_THRESHOLD = 0.05
DEFAULT_TEXT_THRESHOLD = 0.3  # unused by OWLv2, kept for request-contract compat
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_LABELS = 30
# Align Pillow's own bomb guard with our limit so a crafted image cannot decode
# a buffer larger than we would accept.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_model_state: dict[str, Any] = {}


def _load_model() -> None:
    if _model_state:
        return
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    logger.info("loading owlv2 model %s", MODEL_ID)
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
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
    labels: list[str] | None = Field(default=None, max_length=MAX_LABELS)
    confidence_threshold: float = Field(default=DEFAULT_BOX_THRESHOLD, ge=0.0, le=1.0)
    text_threshold: float = Field(default=DEFAULT_TEXT_THRESHOLD, ge=0.0, le=1.0)
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


app = FastAPI(title="vision-grounding-dino", version="2.0.0")
_metrics = build_vision_metrics("vision-grounding-dino")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "vision-grounding-dino",
        "model_id": MODEL_ID,
        "loaded": bool(_model_state),
        "device": _model_state.get("device"),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=_metrics.render(), media_type=VISION_METRICS_CONTENT_TYPE)


def _build_labels(payload: DetectRequest) -> list[str]:
    if payload.labels:
        phrases = [label.strip().lower() for label in payload.labels if label.strip()]
        if not phrases:
            raise HTTPException(status_code=400, detail="labels: at least one non-empty label is required")
        return phrases
    if payload.prompt and payload.prompt.strip():
        phrases = [p.strip().lower() for p in re.split(r"[.,]", payload.prompt) if p.strip()]
        if phrases:
            return phrases
    raise HTTPException(status_code=400, detail="either labels or prompt is required")


@app.post("/generate/detect/grounding-dino", response_model=DetectionResponse)
def detect(payload: DetectRequest) -> DetectionResponse:
    image = _decode_image(payload.image)
    labels = _build_labels(payload)
    _load_model()
    torch = _model_state["torch"]
    processor = _model_state["processor"]
    model = _model_state["model"]
    device = _model_state["device"]

    height, width = image.height, image.width
    with _metrics.observe_inference(MODEL_ID):
        try:
            inputs = processor(text=[labels], images=image, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)

            target_sizes = torch.tensor([[height, width]])
            results = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=float(payload.confidence_threshold),
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("owlv2 inference failed")
            raise HTTPException(status_code=500, detail="detection inference failed") from exc

    if not results:
        logger.warning("owlv2 post-process returned no result container")
        raise HTTPException(status_code=502, detail="detection produced no result container")
    result = results[0]

    scores = result.get("scores")
    label_indices = result.get("labels")
    boxes = result.get("boxes")
    if scores is None or label_indices is None or boxes is None:
        logger.warning("owlv2 post-process missing keys=%s", list(result.keys()))
        raise HTTPException(status_code=502, detail="detection result missing expected keys")

    detections: list[Detection] = []
    for score, label_index, box in zip(scores.tolist(), label_indices.tolist(), boxes.tolist()):
        if not (0 <= int(label_index) < len(labels)):
            continue
        label_name = labels[int(label_index)]
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
                label=label_name,
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


@app.exception_handler(HTTPException)
def _http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_type": "DetectionError", "message": exc.detail},
    )
