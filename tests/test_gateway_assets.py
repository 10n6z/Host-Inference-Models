import base64
import hashlib
import hmac
import importlib
import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_DIR = REPO_ROOT / "model-gateway"
SERVICES_DIR = REPO_ROOT / "services"
SECRET = "gateway-assets-test-secret-with-sufficient-length"


def _b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _token(**overrides):
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": "sw4e-control-plane",
        "aud": "model-gateway",
        "exp": int(time.time()) + 300,
        "userId": "user-a",
        "tenantId": "tenant-a",
        "permittedTasks": ["ocr"],
        "permittedModelIds": ["pp-ocr-v4"],
        **overrides,
    }
    signed = f"{_b64(json.dumps(header, separators=(',', ':')).encode())}.{_b64(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(SECRET.encode(), signed.encode(), hashlib.sha256).digest()
    return f"{signed}.{_b64(signature)}"


def _authorized_headers(**claims):
    return {"Authorization": f"Bearer {_token(**claims)}"}


@pytest.fixture
def gateway_module(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(GATEWAY_DIR))
    monkeypatch.syspath_prepend(str(SERVICES_DIR))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(GATEWAY_DIR / "registry.yaml"))
    monkeypatch.setenv("MODEL_GATEWAY_JWT_SECRET", SECRET)
    monkeypatch.setenv("GATEWAY_ASSETS_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("GATEWAY_MAX_ASSET_BYTES", "16")
    for name in ("main", "assets", "tenant_store"):
        if name in sys.modules:
            del sys.modules[name]
    return importlib.import_module("main")


@pytest.fixture
def client(gateway_module):
    return TestClient(gateway_module.app)


def test_upload_asset_requires_a_gateway_token(client):
    response = client.post("/assets", files={"file": ("a.png", io.BytesIO(b"data"), "image/png")})

    assert response.status_code == 401
    assert response.json()["code"] == "GATEWAY_TOKEN_INVALID"


def test_upload_asset_stores_content_and_reports_checksum(client):
    content = b"tiny-fixture"
    response = client.post(
        "/assets",
        headers=_authorized_headers(),
        files={"file": ("a.png", io.BytesIO(content), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["byte_length"] == len(content)
    assert body["sha256"] == hashlib.sha256(content).hexdigest()


def test_upload_asset_over_the_size_limit_is_rejected(client):
    response = client.post(
        "/assets",
        headers=_authorized_headers(),
        files={"file": ("a.png", io.BytesIO(b"x" * 64), "image/png")},
    )

    assert response.status_code == 413


def test_tenant_cannot_read_another_tenants_asset(client):
    upload = client.post(
        "/assets",
        headers=_authorized_headers(),
        files={"file": ("a.png", io.BytesIO(b"data"), "image/png")},
    )
    asset_id = upload.json()["id"]

    response = client.get(
        f"/assets/{asset_id}",
        headers=_authorized_headers(tenantId="tenant-b"),
    )

    assert response.status_code == 404


def test_jobs_endpoint_requires_a_gateway_token(client):
    response = client.get("/jobs/some-job")

    assert response.status_code == 401
    assert response.json()["code"] == "GATEWAY_TOKEN_INVALID"


def test_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist", headers=_authorized_headers())

    assert response.status_code == 404


def test_tenant_cannot_read_another_tenants_job(client, gateway_module):
    gateway_module.record_tenant_job(
        gateway_module.TenantGatewayJob(
            id="job-a",
            tenant_id="tenant-a",
            model="pp-ocr-v4",
            status="running",
            model_job_id=None,
            response={"status": "running"},
            updated_at="2026-07-31T00:00:00Z",
        )
    )

    same_tenant = client.get("/jobs/job-a", headers=_authorized_headers())
    other_tenant = client.get("/jobs/job-a", headers=_authorized_headers(tenantId="tenant-b"))

    assert same_tenant.status_code == 200
    assert same_tenant.json()["status"] == "running"
    assert other_tenant.status_code == 404
