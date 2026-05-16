import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    servers_dir = Path(__file__).resolve().parents[1] / "servers"
    monkeypatch.syspath_prepend(str(servers_dir))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://testserver")

    if "model_server" in sys.modules:
        del sys.modules["model_server"]

    module = importlib.import_module("model_server")
    module = importlib.reload(module)

    captured = {
        "flux": None,
        "sd35": None,
        "auraflow": None,
        "openflux": None,
        "kokoro": None,
    }

    def _fake_image(model_key):
        def _run(**kwargs):
            captured[model_key] = kwargs
            Path(kwargs["output_path"]).write_bytes(b"PNG")
            return kwargs["output_path"]

        return _run

    def _fake_kokoro(**kwargs):
        captured["kokoro"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": kwargs.get("sample_rate", 24000), "duration_seconds": 1.23}

    monkeypatch.setattr(module.flux_runner, "generate", _fake_image("flux"))
    monkeypatch.setattr(module.sd35_runner, "generate", _fake_image("sd35"))
    monkeypatch.setattr(module.auraflow_runner, "generate", _fake_image("auraflow"))
    monkeypatch.setattr(module.openflux_runner, "generate", _fake_image("openflux"))
    monkeypatch.setattr(module.kokoro_runner, "generate", _fake_kokoro)

    module._captured = captured
    return module


@pytest.fixture
def client(app_module):
    return TestClient(app_module.app)


def test_models_returns_supported_fields(client):
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data

    flux = next(model for model in data["models"] if model["id"] == "flux-1-schnell")
    assert "fields" in flux
    assert "prompt" in flux["fields"]
    assert flux["fields"]["steps"]["default"] == 4


def test_flux_accepts_configurable_params(client, app_module):
    payload = {
        "prompt": "hello",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "guidance_scale": 0,
        "seed": 42,
        "random_seed": False,
    }
    res = client.post("/generate/image/flux", json=payload)
    assert res.status_code == 200

    body = res.json()
    assert body["success"] is True
    assert body["parameters_used"]["seed"] == 42
    assert app_module._captured["flux"]["width"] == 1024
    assert app_module._captured["flux"]["steps"] == 4


def test_invalid_width_rejected(client):
    res = client.post("/generate/image/flux", json={"prompt": "hi", "width": 513})
    assert res.status_code == 422
    body = res.json()
    assert body["success"] is False
    assert body["code"] == "VALIDATION_ERROR"


def test_invalid_steps_rejected(client):
    res = client.post("/generate/image/flux", json={"prompt": "hi", "steps": 999})
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "VALIDATION_ERROR"


def test_deterministic_seed_path(client, app_module):
    res = client.post(
        "/generate/image/sd35",
        json={"prompt": "robot", "seed": 123, "random_seed": False},
    )
    assert res.status_code == 200
    assert app_module._captured["sd35"]["seed"] == 123


def test_random_seed_path(client):
    res = client.post(
        "/generate/image/sd35",
        json={"prompt": "robot", "seed": 123, "random_seed": True},
    )
    assert res.status_code == 200
    used_seed = res.json()["parameters_used"]["seed"]
    assert isinstance(used_seed, int)
    assert used_seed >= 0


def test_unsupported_field_rejected(client):
    res = client.post("/generate/image/openflux", json={"prompt": "x", "negative_prompt": "bad"})
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "UNSUPPORTED_PARAMETER"


def test_tts_accepts_text_voice_language_speed(client, app_module):
    res = client.post(
        "/generate/tts/kokoro",
        json={
            "text": "hello world",
            "voice": "af_heart",
            "language": "en",
            "speed": 1.0,
            "format": "wav",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["parameters_used"]["language"] == "en"
    assert app_module._captured["kokoro"]["lang_code"] == "a"


def test_output_metadata_includes_parameters_used(client):
    res = client.post("/generate/image/auraflow", json={"prompt": "paper craft"})
    assert res.status_code == 200
    body = res.json()
    assert "parameters_used" in body
    assert body["parameters_used"]["prompt"] == "paper craft"


def test_prompt_only_back_compat(client):
    res = client.post("/generate/image/openflux", json={"prompt": "legacy payload"})
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["parameters_used"]["prompt"] == "legacy payload"
