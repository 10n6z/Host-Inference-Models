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
        "fish_speech": None,
        "cosyvoice2": None,
        "indextts2": None,
        "stable_audio_open": None,
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

    def _fake_audio(model_key):
        def _run(**kwargs):
            captured[model_key] = kwargs
            Path(kwargs["output_path"]).write_bytes(b"WAV")
            return {"sample_rate": kwargs.get("sample_rate", 44100), "audio_duration_seconds": 2.34}

        return _run

    monkeypatch.setattr(module.flux_runner, "generate", _fake_image("flux"))
    monkeypatch.setattr(module.sd35_runner, "generate", _fake_image("sd35"))
    monkeypatch.setattr(module.auraflow_runner, "generate", _fake_image("auraflow"))
    monkeypatch.setattr(module.openflux_runner, "generate", _fake_image("openflux"))
    monkeypatch.setattr(module.kokoro_runner, "generate", _fake_kokoro)
    monkeypatch.setattr(module.fish_speech_runner, "generate", _fake_audio("fish_speech"))
    monkeypatch.setattr(module.cosyvoice2_runner, "generate", _fake_audio("cosyvoice2"))
    monkeypatch.setattr(module.indextts2_runner, "generate", _fake_audio("indextts2"))
    monkeypatch.setattr(module.stable_audio_open_runner, "generate", _fake_audio("stable_audio_open"))

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
    stable_audio = next(model for model in data["models"] if model["id"] == "stable-audio-open-1.0")
    assert stable_audio["audioKind"] == "text-to-audio"
    assert stable_audio["fields"]["duration_seconds"]["max"] == 47


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


def test_audio_contract_rejects_arbitrary_gpu_endpoint_field(client):
    res = client.post(
        "/generate/tts/fish-speech",
        json={
            "text": "x",
            "language": "en",
            "voice": "default",
            "gpuEndpoint": "http://evil.example/generate",
        },
    )
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
    assert body["audio_kind"] == "tts"
    assert body["audio_duration_seconds"] == pytest.approx(1.23, rel=1e-6)


def test_fish_speech_contract_and_metadata(client, app_module):
    res = client.post(
        "/generate/tts/fish-speech",
        json={
            "text": "hello fish speech",
            "language": "en",
            "voice": "default",
            "speed": 1.0,
            "format": "wav",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model_id"] == "fish-speech-v1.5"
    assert body["audio_kind"] == "tts"
    assert app_module._captured["fish_speech"]["voice"] == "default"


def test_cosyvoice2_accepts_instruction_and_stream(client, app_module):
    res = client.post(
        "/generate/tts/cosyvoice2",
        json={
            "text": "hello cosyvoice two",
            "language": "en",
            "speaker": "default",
            "instruction": "calm and warm",
            "speed": 1.0,
            "format": "wav",
            "stream": False,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model_id"] == "cosyvoice2-0.5b"
    assert app_module._captured["cosyvoice2"]["instruction"] == "calm and warm"


def test_indextts2_accepts_emotion_and_duration_control(client, app_module):
    res = client.post(
        "/generate/tts/indextts2",
        json={
            "text": "hello index tts",
            "speaker": "default",
            "emotion": "neutral",
            "duration_control": 1.0,
            "speed": 1.0,
            "format": "wav",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model_id"] == "indextts-2"
    assert app_module._captured["indextts2"]["emotion"] == "neutral"


def test_stable_audio_open_seed_mapping(client, app_module):
    res = client.post(
        "/generate/audio/stable-audio-open",
        json={
            "prompt": "ambient cinematic bells",
            "duration_seconds": 10,
            "steps": 50,
            "guidance_scale": 7.0,
            "seed": 42,
            "random_seed": False,
            "format": "wav",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model_id"] == "stable-audio-open-1.0"
    assert body["audio_kind"] == "text-to-audio"
    assert app_module._captured["stable_audio_open"]["seed"] == 42


def test_stable_audio_duration_limit_enforced(client):
    res = client.post(
        "/generate/audio/stable-audio-open",
        json={"prompt": "too long", "duration_seconds": 99},
    )
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "VALIDATION_ERROR"




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
