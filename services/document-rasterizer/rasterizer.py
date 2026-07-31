"""Bounded PDF/TIFF rasterization -- isolated from the public API process.

Re-enforces the page/pixel limits the control-plane already checked at
upload time (documentValidation.ts), because this service is the one that
actually decodes untrusted bytes and cannot rely on a boundary it doesn't
control staying correct forever.

Document-level failures (can't open the file, page count over the limit)
abort the whole document -- there is nothing to salvage. Page-level
failures (one corrupt page in an otherwise valid document) are caught per
page so the caller can preserve every other page, matching plan Task 14's
partial_success requirement.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import pypdfium2 as pdfium
from PIL import Image

MAX_PAGES = 200
MAX_PIXELS = 25_000_000
MAX_RENDER_SECONDS_PER_PAGE = 30
RENDER_SCALE = 2.0  # ~144 DPI at a 72-DPI page unit


class RasterizerError(Exception):
    """Document-level failure -- no pages can be salvaged."""

    def __init__(self, code: str, message: str | None = None):
        super().__init__(message or code)
        self.code = code


@dataclass
class RasterizedPage:
    page_index: int
    width: int
    height: int
    png_bytes: bytes


@dataclass
class PageError:
    page_index: int
    code: str
    message: str


PageResult = Union[RasterizedPage, PageError]


def _encode_bounded(width: int, height: int, image: Image.Image, page_index: int) -> RasterizedPage:
    if width * height > MAX_PIXELS:
        raise RasterizerError("IMAGE_PIXEL_LIMIT", f"Page {page_index} exceeds the 25 MP limit")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return RasterizedPage(page_index=page_index, width=width, height=height, png_bytes=buffer.getvalue())


def rasterize_pdf(path: Path) -> Iterator[PageResult]:
    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise RasterizerError("DOCUMENT_HEADER_INVALID", "Malformed PDF") from exc

    page_count = len(document)
    if page_count > MAX_PAGES:
        raise RasterizerError("DOCUMENT_PAGE_LIMIT", "PDF exceeds the 200-page limit")
    if page_count < 1:
        raise RasterizerError("DOCUMENT_HEADER_INVALID", "PDF has no pages")

    for page_index in range(page_count):
        try:
            started = time.monotonic()
            page = document[page_index]
            width = round(page.get_size()[0] * RENDER_SCALE)
            height = round(page.get_size()[1] * RENDER_SCALE)
            if width * height > MAX_PIXELS:
                raise RasterizerError("IMAGE_PIXEL_LIMIT", f"Page {page_index} exceeds the 25 MP limit")
            bitmap = page.render(scale=RENDER_SCALE)
            image = bitmap.to_pil()
            if time.monotonic() - started > MAX_RENDER_SECONDS_PER_PAGE:
                raise RasterizerError(
                    "DOCUMENT_RENDER_TIMEOUT", f"Page {page_index} exceeded the render time limit"
                )
            yield _encode_bounded(image.width, image.height, image, page_index)
        except RasterizerError as exc:
            yield PageError(page_index=page_index, code=exc.code, message=str(exc))
        except pdfium.PdfiumError as exc:
            yield PageError(page_index=page_index, code="PAGE_RENDER_FAILED", message=str(exc))


def rasterize_tiff(path: Path) -> Iterator[PageResult]:
    try:
        image = Image.open(path)
        page_count = getattr(image, "n_frames", 1)
    except (OSError, ValueError) as exc:
        raise RasterizerError("DOCUMENT_HEADER_INVALID", "Malformed TIFF") from exc

    if page_count > MAX_PAGES:
        raise RasterizerError("DOCUMENT_PAGE_LIMIT", "TIFF exceeds the 200-page limit")
    if page_count < 1:
        raise RasterizerError("DOCUMENT_HEADER_INVALID", "TIFF has no pages")

    for page_index in range(page_count):
        try:
            image.seek(page_index)
            frame = image.convert("RGB")
            yield _encode_bounded(frame.width, frame.height, frame, page_index)
        except RasterizerError as exc:
            yield PageError(page_index=page_index, code=exc.code, message=str(exc))
        except (OSError, ValueError) as exc:
            yield PageError(page_index=page_index, code="PAGE_RENDER_FAILED", message=str(exc))
