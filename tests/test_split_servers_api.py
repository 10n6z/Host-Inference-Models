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


def test_image_server_models_are_image_only(tmp_path, monkeypatch):
    module = _import_server("image_server", tmp_path, monkeypatch)
    client = TestClient(module.app)

    res = client.get("/models")
    assert res.status_code == 200
    models = res.json()["models"]
    assert {model["id"] for model in models} == {
        "flux-1-schnell",
        "stable-diffusion-3.5-medium",
        "auraflow-v0.3",
        "openflux-1",
    }
    assert all(model["modality"] == "image" for model in models)


def test_image_server_flux_response_uses_image_base_url(tmp_path, monkeypatch):
    module = _import_server("image_server", tmp_path, monkeypatch)
    captured = {}

    def _fake_flux(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"PNG")
        return kwargs["output_path"]

    monkeypatch.setattr(module.flux_runner, "generate", _fake_flux)
    client = TestClient(module.app)

    res = client.post(
        "/generate/image/flux",
        json={"prompt": "hello", "seed": 7, "random_seed": False},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["modelId"] == "flux-1-schnell"
    assert body["outputType"] == "image"
    assert body["outputUrl"].startswith("http://image.test:8001/outputs/images/img_")
    assert captured["seed"] == 7


def test_audio_server_models_are_audio_only(tmp_path, monkeypatch):
    module = _import_server("audio_server", tmp_path, monkeypatch)
    client = TestClient(module.app)

    res = client.get("/models")
    assert res.status_code == 200
    models = res.json()["models"]
    assert {model["id"] for model in models} == {
        "kokoro-82m",
        "cosyvoice2-0.5b",
        "fish-speech-v1.5",
        "indextts-2",
        "stable-audio-open-1.0",
    }
    assert {model["modality"] for model in models} == {"text-to-speech", "text-to-audio"}
    assert next(model for model in models if model["id"] == "stable-audio-open-1.0")["endpoint"] == "/generate/audio/stable-audio"


def test_audio_server_kokoro_response_uses_audio_base_url(tmp_path, monkeypatch):
    module = _import_server("audio_server", tmp_path, monkeypatch)
    captured = {}

    def _fake_kokoro(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 24000, "duration_seconds": 1.25}

    monkeypatch.setattr(module.kokoro_runner, "generate", _fake_kokoro)
    client = TestClient(module.app)

    res = client.post("/generate/tts/kokoro", json={"text": "hello", "language": "en"})
    assert res.status_code == 200
    body = res.json()
    assert body["modelId"] == "kokoro-82m"
    assert body["modality"] == "text-to-speech"
    assert body["outputType"] == "audio"
    assert body["outputId"].startswith("tts_")
    assert body["outputUrl"].startswith("http://audio.test:8002/outputs/audio/tts_")
    assert captured["lang_code"] == "a"


def test_audio_server_cosyvoice2_endpoint(tmp_path, monkeypatch):
    module = _import_server("audio_server", tmp_path, monkeypatch)
    captured = {}

    def _fake_cosyvoice2(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 22050, "duration_seconds": 1.0}

    monkeypatch.setattr(module.cosyvoice2_runner, "generate", _fake_cosyvoice2)
    client = TestClient(module.app)

    res = client.post(
        "/generate/tts/cosyvoice2",
        json={"text": "Hello, this is a CosyVoice2 smoke test.", "seed": 42},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["modelId"] == "cosyvoice2-0.5b"
    assert body["modality"] == "text-to-speech"
    assert body["outputId"].startswith("cosyvoice2_")
    assert body["outputUrl"].startswith("http://audio.test:8002/outputs/audio/cosyvoice2_")
    assert captured["seed"] == 42
    assert captured["parameters"]["seed"] == 42


def test_audio_server_new_tts_failure_shape(tmp_path, monkeypatch):
    module = _import_server("audio_server", tmp_path, monkeypatch)

    def _fail_cosyvoice2(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(module.cosyvoice2_runner, "generate", _fail_cosyvoice2)
    client = TestClient(module.app)

    res = client.post("/generate/tts/cosyvoice2", json={"text": "hello"})
    assert res.status_code == 500
    body = res.json()
    assert body == {
        "success": False,
        "code": "TTS_GENERATION_FAILED",
        "message": "synthetic failure",
        "details": {},
    }


def test_audio_server_fish_speech_endpoint(tmp_path, monkeypatch):
    module = _import_server("audio_server", tmp_path, monkeypatch)
    captured = {}

    def _fake_fish_speech(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 44100, "duration_seconds": 1.0}

    monkeypatch.setattr(module.fish_speech_runner, "generate", _fake_fish_speech)
    client = TestClient(module.app)

    res = client.post(
        "/generate/tts/fish-speech",
        json={"text": "Hello, this is a Fish Speech smoke test.", "seed": 42},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["modelId"] == "fish-speech-v1.5"
    assert body["modality"] == "text-to-speech"
    assert body["outputId"].startswith("fish_")
    assert body["outputUrl"].startswith("http://audio.test:8002/outputs/audio/fish_")
    assert captured["seed"] == 42
    assert captured["parameters"]["seed"] == 42


def test_audio_server_indextts2_endpoint(tmp_path, monkeypatch):
    module = _import_server("audio_server", tmp_path, monkeypatch)
    captured = {}

    def _fake_indextts2(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 24000, "duration_seconds": 1.0}

    monkeypatch.setattr(module.indextts2_runner, "generate", _fake_indextts2)
    client = TestClient(module.app)

    res = client.post(
        "/generate/tts/indextts2",
        json={"text": "Hello, this is an IndexTTS two smoke test.", "seed": 42},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["modelId"] == "indextts-2"
    assert body["modality"] == "text-to-speech"
    assert body["outputId"].startswith("indextts2_")
    assert body["outputUrl"].startswith("http://audio.test:8002/outputs/audio/indextts2_")
    assert captured["seed"] == 42
    assert captured["parameters"]["seed"] == 42


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
