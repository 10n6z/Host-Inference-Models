from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from rasterizer import PageError, RasterizedPage, RasterizerError, rasterize_pdf, rasterize_tiff


def _pdf_fixture(tmp_path: Path, pages: int, page_size=(200, 300)) -> Path:
    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(*page_size)
    path = tmp_path / "fixture.pdf"
    document.save(str(path))
    return path


def _tiff_fixture(tmp_path: Path, pages: int, size=(50, 50)) -> Path:
    images = [Image.new("RGB", size, color=(i * 10, 0, 0)) for i in range(pages)]
    path = tmp_path / "fixture.tiff"
    images[0].save(path, format="TIFF", save_all=True, append_images=images[1:])
    return path


def test_rasterize_pdf_renders_every_page_in_order(tmp_path):
    path = _pdf_fixture(tmp_path, pages=3)

    results = list(rasterize_pdf(path))

    assert [r.page_index for r in results] == [0, 1, 2]
    assert all(isinstance(r, RasterizedPage) for r in results)
    assert all(r.png_bytes.startswith(b"\x89PNG") for r in results)


def test_rasterize_pdf_rejects_a_document_over_the_page_limit(tmp_path, monkeypatch):
    import rasterizer

    monkeypatch.setattr(rasterizer, "MAX_PAGES", 2)
    path = _pdf_fixture(tmp_path, pages=3)

    with pytest.raises(RasterizerError) as exc_info:
        list(rasterize_pdf(path))
    assert exc_info.value.code == "DOCUMENT_PAGE_LIMIT"


def test_rasterize_pdf_rejects_a_malformed_file(tmp_path):
    path = tmp_path / "not-a-pdf.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(RasterizerError) as exc_info:
        list(rasterize_pdf(path))
    assert exc_info.value.code == "DOCUMENT_HEADER_INVALID"


def test_rasterize_pdf_enforces_the_pixel_limit_per_page(tmp_path, monkeypatch):
    import rasterizer

    monkeypatch.setattr(rasterizer, "MAX_PIXELS", 100)
    path = _pdf_fixture(tmp_path, pages=1)

    results = list(rasterize_pdf(path))

    assert len(results) == 1
    assert isinstance(results[0], PageError)
    assert results[0].code == "IMAGE_PIXEL_LIMIT"


def test_rasterize_tiff_renders_every_page_in_order(tmp_path):
    path = _tiff_fixture(tmp_path, pages=3)

    results = list(rasterize_tiff(path))

    assert [r.page_index for r in results] == [0, 1, 2]
    assert all(isinstance(r, RasterizedPage) for r in results)


def test_rasterize_tiff_rejects_a_malformed_file(tmp_path):
    path = tmp_path / "not-a-tiff.tiff"
    path.write_bytes(b"not a tiff")

    with pytest.raises(RasterizerError) as exc_info:
        list(rasterize_tiff(path))
    assert exc_info.value.code == "DOCUMENT_HEADER_INVALID"


def test_rasterize_tiff_rejects_a_document_over_the_page_limit(tmp_path, monkeypatch):
    import rasterizer

    monkeypatch.setattr(rasterizer, "MAX_PAGES", 2)
    path = _tiff_fixture(tmp_path, pages=3)

    with pytest.raises(RasterizerError) as exc_info:
        list(rasterize_tiff(path))
    assert exc_info.value.code == "DOCUMENT_PAGE_LIMIT"
