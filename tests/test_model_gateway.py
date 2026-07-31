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
GATEWAY_SECRET = "gateway-model-gateway-test-secret-with-sufficient-length"


def _b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


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
    signature = hmac.new(
        GATEWAY_SECRET.encode(), signed.encode(), hashlib.sha256
    ).digest()
    return TestClient(
        gateway.app,
        headers={"Authorization": f"Bearer {signed}.{_b64(signature)}"},
    )


def _import_gateway_module(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(GATEWAY_DIR))
    monkeypatch.syspath_prepend(str(SERVICES_DIR))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(GATEWAY_DIR / "registry.yaml"))
    monkeypatch.setenv("MODEL_GATEWAY_JWT_SECRET", GATEWAY_SECRET)
    if "main" in sys.modules:
        del sys.modules["main"]
    return importlib.import_module("main")


def _import_kokoro_service(tmp_path, monkeypatch):
    service_dir = SERVICES_DIR / "audio-kokoro"
    monkeypatch.syspath_prepend(str(SERVICES_DIR))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "servers"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache" / "hub"))
    monkeypatch.setenv("AUDIO_PUBLIC_BASE_URL", "http://audio-kokoro.test:8102")

    fake_kokoro = type(sys)("kokoro")

    class _FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return iter([])

    fake_kokoro.KPipeline = _FakePipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    for module_name in list(sys.modules):
        if module_name in {"main", "common.audio_service_base"} or module_name.startswith("runners."):
            del sys.modules[module_name]

    import importlib.util

    spec = importlib.util.spec_from_file_location("kokoro_main", service_dir / "main.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gateway_health_and_models(monkeypatch, tmp_path):
    gateway = _import_gateway_module(monkeypatch, tmp_path)
    client = _gateway_client(gateway)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["service"] == "model-gateway"

    models = client.get("/models")
    assert models.status_code == 200
    model_ids = {entry["id"] for entry in models.json()["models"]}
    assert "kokoro-82m" in model_ids
    assert "melotts" in model_ids
    assert "espnet-vits" in model_ids
    assert "bark-small" in model_ids
    assert "cosyvoice2-0.5b" in model_ids


def test_gateway_unknown_model_returns_contract(monkeypatch, tmp_path):
    gateway = _import_gateway_module(monkeypatch, tmp_path)
    client = _gateway_client(gateway)

    response = client.post("/generate", json={"model": "does-not-exist", "text": "hello"})
    assert response.status_code == 404
    body = response.json()
    assert body["status"] == "failed"
    assert body["model"] == "does-not-exist"
    assert body["error"]["type"] == "UnknownModel"
    assert "request_id" in body["metadata"]


def test_gateway_routes_kokoro_to_family_service(monkeypatch, tmp_path):
    gateway = _import_gateway_module(monkeypatch, tmp_path)
    client = _gateway_client(gateway)

    upstream_payload = {
        "success": True,
        "status": "completed",
        "model_id": "kokoro-82m",
        "output_url": "/outputs/audio/tts_test.wav",
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
            "/generate",
            json={"model": "kokoro-82m", "text": "hello", "language": "en"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "kokoro-82m"
    assert body["metadata"]["family"] == "audio-kokoro"
    mock_client.post.assert_awaited_once()
    called_url = mock_client.post.await_args.args[0]
    called_json = mock_client.post.await_args.kwargs["json"]
    assert called_url == "http://audio-kokoro:8102/generate"
    assert called_json["model"] == "kokoro-82m"
    assert called_json["text"] == "hello"


def test_gateway_routes_cosyvoice_without_model_field(monkeypatch, tmp_path):
    gateway = _import_gateway_module(monkeypatch, tmp_path)
    client = _gateway_client(gateway)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"success": True, "modelId": "cosyvoice2-0.5b"}

    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/generate",
            json={"model": "cosyvoice2-0.5b", "text": "hello", "seed": 42},
        )

    assert response.status_code == 200
    called_url = mock_client.post.await_args.args[0]
    called_json = mock_client.post.await_args.kwargs["json"]
    assert called_url == "http://audio-cosyvoice:8113/generate"
    assert "model" not in called_json
    assert called_json["text"] == "hello"
    assert called_json["seed"] == 42


def test_gateway_flattens_documented_nested_contract(monkeypatch, tmp_path):
    gateway = _import_gateway_module(monkeypatch, tmp_path)
    client = _gateway_client(gateway)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"success": True, "model_id": "kokoro-82m"}

    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/generate",
            json={
                "model": "kokoro-82m",
                "input": {"text": "hello"},
                "parameters": {"voice": "af_heart", "speed": 1.1},
                "request_id": "nested-smoke",
            },
        )

    assert response.status_code == 200
    called_json = mock_client.post.await_args.kwargs["json"]
    called_headers = mock_client.post.await_args.kwargs["headers"]
    assert called_json == {
        "text": "hello",
        "voice": "af_heart",
        "speed": 1.1,
        "model": "kokoro-82m",
    }
    assert called_headers["X-Request-ID"] == "nested-smoke"


def test_kokoro_service_generate_with_mock_runner(tmp_path, monkeypatch):
    module = _import_kokoro_service(tmp_path, monkeypatch)
    captured = {}

    def _fake_kokoro(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 24000, "duration_seconds": 1.0}

    monkeypatch.setattr(module.kokoro_runner, "generate", _fake_kokoro)
    client = TestClient(module.app)

    response = client.post(
        "/generate",
        json={"model": "kokoro-82m", "text": "hello", "language": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "kokoro-82m"
    assert body["metadata"]["family"] == "audio-kokoro"
    assert captured["lang_code"] == "a"


def test_gateway_routes_espnet_to_family_service(monkeypatch, tmp_path):
    gateway = _import_gateway_module(monkeypatch, tmp_path)
    client = _gateway_client(gateway)

    upstream_payload = {
        "success": True,
        "status": "completed",
        "model_id": "espnet-vits",
        "output_url": "/outputs/audio/espnet_test.wav",
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
            "/generate",
            json={"model": "espnet-vits", "text": "hello"},
        )

    assert response.status_code == 200
    called_url = mock_client.post.await_args.args[0]
    called_json = mock_client.post.await_args.kwargs["json"]
    assert called_url == "http://audio-espnet:8103/generate"
    assert called_json["model"] == "espnet-vits"


def test_gateway_routes_bark_to_family_service(monkeypatch, tmp_path):
    gateway = _import_gateway_module(monkeypatch, tmp_path)
    client = _gateway_client(gateway)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"success": True, "model_id": "bark-small"}

    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/generate",
            json={"model": "bark-small", "text": "hello"},
        )

    assert response.status_code == 200
    called_url = mock_client.post.await_args.args[0]
    called_json = mock_client.post.await_args.kwargs["json"]
    assert called_url == "http://audio-bark:8106/generate"
    assert called_json["model"] == "bark-small"


def test_kokoro_service_rejects_unknown_model(tmp_path, monkeypatch):
    module = _import_kokoro_service(tmp_path, monkeypatch)
    client = TestClient(module.app)

    response = client.post("/generate", json={"model": "melotts", "text": "hello"})
    assert response.status_code == 404
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]["type"] == "UnknownModel"
