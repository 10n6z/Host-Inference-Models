import base64
import hashlib
import hmac
import importlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_DIR = REPO_ROOT / "model-gateway"
SERVICES_DIR = REPO_ROOT / "services"
TEST_SIGNING_KEY = "gateway-vllm-adapter-test-key-with-sufficient-length"


def _b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _import_gateway_module(monkeypatch):
    monkeypatch.syspath_prepend(str(GATEWAY_DIR))
    monkeypatch.syspath_prepend(str(SERVICES_DIR))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(GATEWAY_DIR / "registry.yaml"))
    monkeypatch.setenv("MODEL_GATEWAY_JWT_SECRET", TEST_SIGNING_KEY)
    if "main" in sys.modules:
        del sys.modules["main"]
    return importlib.import_module("main")


def _gateway_client(gateway):
    models = gateway._models_payload()
    payload = {
        "iss": "sw4e-control-plane",
        "aud": "model-gateway",
        "exp": int(time.time()) + 300,
        "userId": "test-user",
        "tenantId": "test-tenant",
        "permittedTasks": sorted({model["task"] for model in models}),
        "permittedModelIds": sorted({model["id"] for model in models}),
    }
    header = {"alg": "HS256", "typ": "JWT"}
    signed = f"{_b64(json.dumps(header, separators=(',', ':')).encode())}.{_b64(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(TEST_SIGNING_KEY.encode(), signed.encode(), hashlib.sha256).digest()
    token = f"{signed}.{_b64(signature)}"
    client = TestClient(gateway.app)
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


def _mock_chat_completion(content: str, finish_reason: str = "stop"):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {
        "choices": [
            {"message": {"content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    return mock_response


def test_vision_planner_is_listed_as_a_text_generation_model(monkeypatch):
    gateway = _import_gateway_module(monkeypatch)
    client = _gateway_client(gateway)

    response = client.get("/models")

    model_ids = {entry["id"] for entry in response.json()["models"]}
    assert "vision-planner" in model_ids


def test_generate_routes_to_vllm_chat_completions_and_extracts_output_text(monkeypatch):
    gateway = _import_gateway_module(monkeypatch)
    client = _gateway_client(gateway)

    with patch("adapters.vllm.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = _mock_chat_completion(" OK.")
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/generate",
            json={
                "model": "vision-planner",
                "input": {"prompt": "Reply with only the word: OK"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == " OK."
    assert body["metadata"]["provider"] == "vllm"

    called_url = mock_client.post.await_args.args[0]
    called_json = mock_client.post.await_args.kwargs["json"]
    assert called_url == "http://host.docker.internal:8126/v1/chat/completions"
    assert called_json["model"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert called_json["messages"] == [
        {"role": "user", "content": "Reply with only the word: OK"}
    ]


def test_generate_rejects_missing_prompt_for_vllm(monkeypatch):
    gateway = _import_gateway_module(monkeypatch)
    client = _gateway_client(gateway)

    response = client.post("/generate", json={"model": "vision-planner", "input": {}})

    assert response.status_code == 422


def test_generate_surfaces_vllm_service_unavailable_as_503(monkeypatch):
    gateway = _import_gateway_module(monkeypatch)
    client = _gateway_client(gateway)

    with patch("adapters.vllm.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_response = AsyncMock()
        mock_response.status_code = 503
        mock_response.text = "engine not ready"
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/generate",
            json={"model": "vision-planner", "input": {"prompt": "hello"}},
        )

    assert response.status_code == 503
