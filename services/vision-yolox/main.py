"""vision-yolox — YOLOX-S closed-set (COCO-80) object-detection service.

Contract (matches sandbox's detectorRouting.ts "yolox-s" model id, routed for
jobs with detectorMode="yolox"):

    POST /generate/detect/yolox
    Request JSON:
      { "prompt": "...", "image": "<base64>", "confidence_threshold": 0.25 }
    Response JSON:
      { "detections": [ { "label": "person", "confidence": 0.94,
                          "box": { "x": 20, "y": 30, "width": 150, "height": 300 } } ] }

Coordinates are in the original uploaded image's pixel space. YOLOX-S was
trained on OpenCV-loaded (BGR) images with no channel normalization beyond a
letterbox resize/pad -- see yolox.data.data_augment.preproc upstream, ported
here without the cv2 dependency (PIL resize substituted for cv2.resize).
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from vision_common_metrics import VISION_METRICS_CONTENT_TYPE, build_vision_metrics

logger = logging.getLogger("vision-yolox")
logging.basicConfig(level=os.environ.get("VISION_YOLOX_LOG_LEVEL", "info").upper())

MODEL_NAME = os.environ.get("VISION_YOLOX_MODEL_NAME", "yolox-s")
INPUT_SIZE = (640, 640)
DEFAULT_CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# github.com/Megvii-BaseDetection/YOLOX yolox/data/datasets/coco_classes.py,
# tag 0.3.0 (apache-2.0), inlined to avoid importing yolox.data (which pulls
# pycocotools -- unused by this inference-only service).
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

_model_state: dict[str, Any] = {}


def _load_model() -> None:
    if _model_state:
        return
    import torch
    from yolox.exp import get_exp
    from yolox.utils import postprocess

    logger.info("loading yolox model %s", MODEL_NAME)
    exp = get_exp(exp_name=MODEL_NAME)
    model = exp.get_model()
    checkpoint_url = (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
        "0.1.1rc0/yolox_s.pth"
    )
    checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _model_state["model"] = model
    _model_state["device"] = device
    _model_state["torch"] = torch
    _model_state["postprocess"] = postprocess


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str | None = Field(default=None, max_length=4000)
    image: str = Field(..., min_length=1)
    confidence_threshold: float = Field(default=DEFAULT_CONF_THRESHOLD, ge=0.0, le=1.0)
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


app = FastAPI(title="vision-yolox", version="1.0.0")
_metrics = build_vision_metrics("vision-yolox")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "vision-yolox",
        "model_name": MODEL_NAME,
        "loaded": bool(_model_state),
        "device": _model_state.get("device"),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=_metrics.render(), media_type=VISION_METRICS_CONTENT_TYPE)


def _letterbox_bgr(image: Image.Image, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    # Port of yolox.data.data_augment.preproc: letterbox resize + pad with
    # 114, no normalization beyond that. YOLOX-S was trained on BGR frames
    # from cv2.imread, so the RGB PIL decode is reversed to BGR before this.
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = rgb[:, :, ::-1]
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    ratio = min(input_size[0] / bgr.shape[0], input_size[1] / bgr.shape[1])
    resized_h, resized_w = int(bgr.shape[0] * ratio), int(bgr.shape[1] * ratio)
    resized = np.asarray(
        Image.fromarray(bgr).resize((resized_w, resized_h), Image.BILINEAR)
    )
    padded[:resized_h, :resized_w] = resized
    chw = padded.transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32), ratio


@app.post("/generate/detect/yolox", response_model=DetectionResponse)
def detect(payload: DetectRequest) -> DetectionResponse:
    image = _decode_image(payload.image)
    _load_model()
    torch = _model_state["torch"]
    model = _model_state["model"]
    device = _model_state["device"]
    postprocess = _model_state["postprocess"]

    width, height = image.width, image.height
    tensor_input, ratio = _letterbox_bgr(image, INPUT_SIZE)

    with _metrics.observe_inference(MODEL_NAME):
        try:
            batch = torch.from_numpy(tensor_input).unsqueeze(0).to(device)
            with torch.inference_mode():
                raw_output = model(batch)
                # postprocess() mutates its input tensor in place (upstream
                # yolox.utils.boxes.postprocess); torch forbids that on an
                # inference-mode tensor once outside the inference_mode context,
                # so this call must stay inside the same block as the forward pass.
                results = postprocess(
                    raw_output,
                    len(COCO_CLASSES),
                    conf_thre=float(payload.confidence_threshold),
                    nms_thre=NMS_THRESHOLD,
                )
        except Exception as exc:  # pragma: no cover
            logger.exception("yolox inference failed")
            raise HTTPException(status_code=500, detail="detection inference failed") from exc

    detections: list[Detection] = []
    result = results[0] if results else None
    if result is not None:
        for row in result.tolist():
            x0, y0, x1, y1, obj_conf, class_conf, class_idx = row
            confidence = float(obj_conf) * float(class_conf)
            if not (0.0 <= confidence <= 1.0):
                continue
            label_name = COCO_CLASSES[int(class_idx)] if 0 <= int(class_idx) < len(COCO_CLASSES) else None
            if not label_name:
                continue
            x = max(0.0, min(x0 / ratio, width))
            y = max(0.0, min(y0 / ratio, height))
            w = max(0.0, min(x1 / ratio, width) - x)
            h = max(0.0, min(y1 / ratio, height) - y)
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
