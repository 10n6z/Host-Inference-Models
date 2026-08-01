"""document-rasterizer -- isolated bounded PDF/TIFF page rendering.

Reads a tenant-owned asset by ID from the shared, tenant-scoped asset store
(the same layout model-gateway/assets.py writes to), rasterizes each page
within the enforced limits, and writes rendered pages back into that same
tenant's asset directory rather than returning raw image bytes over JSON.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rasterizer import PageError, RasterizedPage, RasterizerError, rasterize_pdf, rasterize_tiff
from vision_common_metrics import VISION_METRICS_CONTENT_TYPE, build_vision_metrics

logger = logging.getLogger("document-rasterizer")
logging.basicConfig(level=os.environ.get("DOCUMENT_RASTERIZER_LOG_LEVEL", "info").upper())

ASSETS_ROOT = Path(os.environ.get("GATEWAY_ASSETS_ROOT", "/gpt-lab/long/computer-vision/assets"))

app = FastAPI(title="document-rasterizer", version="1.0.0")
_metrics = build_vision_metrics("document-rasterizer")

RASTERIZERS = {
    "application/pdf": rasterize_pdf,
    "image/tiff": rasterize_tiff,
}


class RasterizeRequest(BaseModel):
    tenant_id: str
    asset_id: str
    mime_type: str


class PageStatus(BaseModel):
    page_index: int
    status: str
    width: int | None = None
    height: int | None = None
    path: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class RasterizeResponse(BaseModel):
    status: str
    pages: list[PageStatus]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "document-rasterizer"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=_metrics.render(), media_type=VISION_METRICS_CONTENT_TYPE)


def _resolve_asset_path(tenant_id: str, asset_id: str) -> Path:
    # tenant_id/asset_id come from the authenticated gateway request, never
    # straight from client input, but a defense-in-depth boundary check
    # costs nothing and catches a future caller that skips validation.
    tenant_dir = (ASSETS_ROOT / tenant_id).resolve()
    if not str(tenant_dir).startswith(str(ASSETS_ROOT.resolve())):
        raise HTTPException(status_code=400, detail={"code": "INVALID_TENANT", "message": "Invalid tenant id"})
    asset_path = (tenant_dir / asset_id).resolve()
    if not str(asset_path).startswith(str(tenant_dir)):
        raise HTTPException(status_code=400, detail={"code": "INVALID_ASSET_ID", "message": "Invalid asset id"})
    if not asset_path.is_file():
        raise HTTPException(status_code=404, detail={"code": "ASSET_NOT_FOUND", "message": "Asset not found"})
    return asset_path


@app.post("/rasterize", response_model=RasterizeResponse)
def rasterize(request: RasterizeRequest) -> RasterizeResponse:
    rasterizer = RASTERIZERS.get(request.mime_type)
    if rasterizer is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": f"Unsupported type '{request.mime_type}'"},
        )

    asset_path = _resolve_asset_path(request.tenant_id, request.asset_id)
    output_dir = asset_path.parent / f"{request.asset_id}-pages"
    output_dir.mkdir(parents=True, exist_ok=True)

    with _metrics.observe_inference(request.mime_type):
        try:
            results = list(rasterizer(asset_path))
        except RasterizerError as exc:
            raise HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc)}) from exc

    pages: list[PageStatus] = []
    for result in results:
        if isinstance(result, RasterizedPage):
            page_path = output_dir / f"{result.page_index}.png"
            page_path.write_bytes(result.png_bytes)
            pages.append(
                PageStatus(
                    page_index=result.page_index,
                    status="completed",
                    width=result.width,
                    height=result.height,
                    path=str(page_path),
                )
            )
        elif isinstance(result, PageError):
            pages.append(
                PageStatus(
                    page_index=result.page_index,
                    status="failed",
                    error_code=result.code,
                    error_message=result.message,
                )
            )

    status = "partial_success" if any(page.status == "failed" for page in pages) else "completed"
    return RasterizeResponse(status=status, pages=sorted(pages, key=lambda page: page.page_index))


@app.exception_handler(HTTPException)
def _http_exception_handler(_, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "ERROR", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=detail)
