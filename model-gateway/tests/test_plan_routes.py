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
    ["gateway", "plan", "routes", "test", "key", "with", "sufficient", "length"]
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
        "permittedTasks": ["text"],
        "permittedModelIds": ["commercial-planner-openai"],
    }
    payload.update(overrides)
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(payload).encode())}"
    signature = hmac.new(TEST_SIGNING_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(signature)}"


PLAN_REQUEST_BODY = {
    "prompt": "Read the label from this package.",
    "image_count": 1,
    "requested_domain": "general",
    "requested_tasks": [],
    "ocr_mode": "auto",
    "ocr_language": "auto",
    "detector_mode": "auto",
    "requested_labels": [],
}


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


def test_plan_requires_authentication(client):
    response = client.post("/plan/openai", json=PLAN_REQUEST_BODY)
    assert response.status_code == 401


def test_plan_rejects_an_unknown_provider(client):
    response = client.post(
        "/plan/mistral",
        json=PLAN_REQUEST_BODY,
        headers={"Authorization": f"Bearer {_token()}"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "UNKNOWN_PROVIDER"


def test_plan_proxies_to_commercial_planner_for_openai(client):
    upstream_payload = {
        "domain": "general",
        "tasks": ["ocr", "annotate", "report"],
        "reason": "openai planner reason",
        "source": "openai",
        "warnings": [],
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
            "/plan/openai",
            json=PLAN_REQUEST_BODY,
            headers={"Authorization": f"Bearer {_token()}"},
        )
        called_url = mock_client.post.call_args.args[0]

    assert response.status_code == 200
    assert response.json() == upstream_payload
    assert called_url.endswith("/generate/plan/openai")


def test_plan_surfaces_a_typed_error_from_commercial_planner(client):
    mock_response = AsyncMock()
    mock_response.status_code = 422
    mock_response.json = lambda: {"code": "REFUSAL", "message": "provider declined to plan"}

    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/plan/anthropic",
            json=PLAN_REQUEST_BODY,
            headers={"Authorization": f"Bearer {_token()}"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "REFUSAL"


def test_plan_returns_provider_unavailable_when_commercial_planner_is_unreachable(client):
    import httpx

    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/plan/openai",
            json=PLAN_REQUEST_BODY,
            headers={"Authorization": f"Bearer {_token()}"},
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "PROVIDER_UNAVAILABLE"
