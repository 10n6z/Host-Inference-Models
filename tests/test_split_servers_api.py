import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def _import_server(module_name, tmp_path, monkeypatch):
    servers_dir = Path(__file__).resolve().parents[1] / "servers"
    monkeypatch.syspath_prepend(str(servers_dir))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache" / "hub"))
    monkeypatch.setenv("IMAGE_PUBLIC_BASE_URL", "http://image.test:8001")
    monkeypatch.setenv("AUDIO_PUBLIC_BASE_URL", "http://audio.test:8002")

    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_audio_server_models_are_audio_only(tmp_path, monkeypatch):
    module = _import_server("audio_server", tmp_path, monkeypatch)
    client = TestClient(module.app)

    res = client.get("/models")
    assert res.status_code == 200
    models = res.json()["models"]
    assert {model["id"] for model in models} == {
        "stable-audio-open-1.0",
        "mms-tts",
        "speecht5-tts",
        "f5-tts",
        "e2-tts",
        "kitten-tts",
    }
    assert {model["modality"] for model in models} == {"text-to-speech", "text-to-audio"}
    assert next(model for model in models if model["id"] == "stable-audio-open-1.0")["endpoint"] == "/generate/audio/stable-audio"


def test_audio_server_stable_audio_endpoint(tmp_path, monkeypatch):
    module = _import_server("audio_server", tmp_path, monkeypatch)
    captured = {}

    def _fake_stable_audio(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 44100, "audio_duration_seconds": 2.5}

    monkeypatch.setattr(module.stable_audio_runner, "generate", _fake_stable_audio)
    client = TestClient(module.app)

    res = client.post(
        "/generate/audio/stable-audio",
        json={
            "prompt": "ambient bells",
            "negative_prompt": "noise",
            "duration_seconds": 10,
            "steps": 100,
            "guidance_scale": 7.0,
            "seed": 42,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["modelId"] == "stable-audio-open-1.0"
    assert body["modality"] == "text-to-audio"
    assert body["outputId"].startswith("audio_")
    assert body["outputUrl"].startswith("http://audio.test:8002/outputs/audio/audio_")
    assert captured["seed"] == 42
    assert captured["duration_seconds"] == 10
