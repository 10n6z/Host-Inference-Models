import base64
import hashlib
import hmac
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_DIR = REPO_ROOT / "model-gateway"
SERVICES_DIR = REPO_ROOT / "services"
TEST_SIGNING_KEY = "-".join(
    ["gateway", "document", "routes", "test", "key", "with", "sufficient", "length"]
)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _token(**overrides):
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": "sw4e-control-plane",
        "aud": "model-gateway",
        "exp": int(time.time()) + 300,
        "userId": "user-1",
        "tenantId": "tenant-a",
        "permittedTasks": ["document"],
        "permittedModelIds": ["document-rasterizer"],
    }
    payload.update(overrides)
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(payload).encode())}"
    signature = hmac.new(TEST_SIGNING_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(signature)}"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_JWT_SECRET", TEST_SIGNING_KEY)
    monkeypatch.setenv("GATEWAY_ASSETS_ROOT", str(tmp_path))
    monkeypatch.syspath_prepend(str(GATEWAY_DIR))
    monkeypatch.syspath_prepend(str(SERVICES_DIR))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(GATEWAY_DIR / "registry.yaml"))
    for name in ("main", "assets", "auth", "tenant_store"):
        sys.modules.pop(name, None)
    import main as gateway_main

    return TestClient(gateway_main.app)


def test_rasterize_requires_authentication(client):
    response = client.post("/documents/some-asset/rasterize", json={"mime_type": "application/pdf"})
    assert response.status_code == 401


def test_rasterize_calls_document_rasterizer_and_strips_host_paths(client):
    upstream_payload = {
        "status": "completed",
        "pages": [
            {"page_index": 0, "status": "completed", "width": 100, "height": 200, "path": "/gpt-lab/long/computer-vision/assets/tenant-a/asset-1-pages/0.png"},
        ],
    }
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: upstream_payload

    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/documents/asset-1/rasterize",
            json={"mime_type": "application/pdf"},
            headers={"Authorization": f"Bearer {_token()}"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["pages"] == [{"page_index": 0, "status": "completed", "width": 100, "height": 200, "error_code": None, "error_message": None}]
    assert "path" not in body["pages"][0]


def test_get_page_requires_authentication(client):
    response = client.get("/documents/asset-1/pages/0")
    assert response.status_code == 401


def test_get_page_streams_png_bytes_for_the_owning_tenant(client, tmp_path):
    pages_dir = tmp_path / "tenant-a" / "asset-1-pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "0.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-page-bytes")

    response = client.get(
        "/documents/asset-1/pages/0",
        headers={"Authorization": f"Bearer {_token()}"},
    )
    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n\x1a\nfake-page-bytes"
    assert response.headers["content-type"] == "image/png"


def test_get_page_404s_for_a_different_tenant(client, tmp_path):
    pages_dir = tmp_path / "tenant-a" / "asset-1-pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "0.png").write_bytes(b"fake")

    response = client.get(
        "/documents/asset-1/pages/0",
        headers={"Authorization": f"Bearer {_token(tenantId='tenant-b')}"},
    )
    assert response.status_code == 404
