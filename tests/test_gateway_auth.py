import base64
import hashlib
import hmac
import importlib
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_DIR = REPO_ROOT / "model-gateway"
SERVICES_DIR = REPO_ROOT / "services"
SECRET = "gateway-auth-test-secret-with-sufficient-length"


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


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(GATEWAY_DIR))
    monkeypatch.syspath_prepend(str(SERVICES_DIR))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(GATEWAY_DIR / "registry.yaml"))
    monkeypatch.setenv("MODEL_GATEWAY_JWT_SECRET", SECRET)
    if "main" in sys.modules:
        del sys.modules["main"]
    gateway = importlib.import_module("main")
    return TestClient(gateway.app)


def _authorized_headers(**claims):
    return {"Authorization": f"Bearer {_token(**claims)}"}


def test_models_rejects_missing_gateway_token(client):
    response = client.get("/models")

    assert response.status_code == 401
    assert response.json()["code"] == "GATEWAY_TOKEN_INVALID"


def test_models_only_returns_models_permitted_to_the_tenant(client):
    response = client.get("/models", headers=_authorized_headers())

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["models"]] == ["pp-ocr-v4"]


def test_generate_rejects_a_model_outside_the_token_scope(client):
    response = client.post(
        "/generate",
        headers=_authorized_headers(),
        json={"model": "rtdetr-r50vd", "task": "object-detection", "image": "ignored"},
    )

    assert response.status_code == 403
    assert response.json()["code"] == "GATEWAY_SCOPE_DENIED"


def test_unknown_model_does_not_disclose_models_outside_the_token_scope(client):
    response = client.post(
        "/generate",
        headers=_authorized_headers(),
        json={"model": "not-a-model", "task": "ocr", "image": "ignored"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["details"]["available_models"] == ["pp-ocr-v4"]


def test_models_rejects_an_expired_gateway_token(client):
    response = client.get("/models", headers=_authorized_headers(exp=int(time.time()) - 1))

    assert response.status_code == 401
    assert response.json()["code"] == "GATEWAY_TOKEN_INVALID"
