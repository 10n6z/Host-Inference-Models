import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vision-common"))
os.environ.setdefault("GATEWAY_ASSETS_ROOT", "/tmp/dr-test-assets")

import main  # noqa: E402  (import after env var set)
from rasterizer import PageError, RasterizedPage  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ASSETS_ROOT", tmp_path)
    tenant_dir = tmp_path / "tenant-a"
    tenant_dir.mkdir(parents=True)
    (tenant_dir / "doc-1").write_bytes(b"fixture bytes")
    return TestClient(main.app)


def test_preserves_successful_pages_when_one_page_fails(client, monkeypatch):
    def fake_rasterize_pdf(path):
        yield RasterizedPage(page_index=0, width=10, height=10, png_bytes=b"\x89PNGpage0")
        yield PageError(page_index=1, code="PAGE_RENDER_FAILED", message="corrupt page")
        yield RasterizedPage(page_index=2, width=10, height=10, png_bytes=b"\x89PNGpage2")

    monkeypatch.setitem(main.RASTERIZERS, "application/pdf", fake_rasterize_pdf)

    response = client.post(
        "/rasterize",
        json={"tenant_id": "tenant-a", "asset_id": "doc-1", "mime_type": "application/pdf"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "partial_success"
    assert [p["status"] for p in body["pages"]] == ["completed", "failed", "completed"]
    assert body["pages"][1]["page_index"] == 1
    assert body["pages"][1]["error_code"] == "PAGE_RENDER_FAILED"


def test_all_pages_succeeding_is_completed(client, monkeypatch):
    def fake_rasterize_pdf(path):
        yield RasterizedPage(page_index=0, width=10, height=10, png_bytes=b"\x89PNGpage0")

    monkeypatch.setitem(main.RASTERIZERS, "application/pdf", fake_rasterize_pdf)

    response = client.post(
        "/rasterize",
        json={"tenant_id": "tenant-a", "asset_id": "doc-1", "mime_type": "application/pdf"},
    )

    assert response.json()["status"] == "completed"


def test_rejects_unsupported_media_type(client):
    response = client.post(
        "/rasterize",
        json={"tenant_id": "tenant-a", "asset_id": "doc-1", "mime_type": "image/png"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_rejects_an_asset_id_that_escapes_the_tenant_directory(client):
    response = client.post(
        "/rasterize",
        json={"tenant_id": "tenant-a", "asset_id": "../tenant-b/secret", "mime_type": "application/pdf"},
    )

    assert response.status_code in (400, 404)


def test_rejects_a_missing_asset(client):
    response = client.post(
        "/rasterize",
        json={"tenant_id": "tenant-a", "asset_id": "does-not-exist", "mime_type": "application/pdf"},
    )

    assert response.status_code == 404
    assert response.json()["code"] == "ASSET_NOT_FOUND"


def test_writes_rendered_pages_under_the_same_tenant_directory(client, monkeypatch, tmp_path):
    def fake_rasterize_pdf(path):
        yield RasterizedPage(page_index=0, width=10, height=10, png_bytes=b"\x89PNGpage0")

    monkeypatch.setitem(main.RASTERIZERS, "application/pdf", fake_rasterize_pdf)

    response = client.post(
        "/rasterize",
        json={"tenant_id": "tenant-a", "asset_id": "doc-1", "mime_type": "application/pdf"},
    )

    page_path = Path(response.json()["pages"][0]["path"])
    assert page_path.exists()
    assert page_path.read_bytes() == b"\x89PNGpage0"
    assert str(tmp_path / "tenant-a") in str(page_path)
