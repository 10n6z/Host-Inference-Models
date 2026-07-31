"""Contract tests for the active combined-server text-to-speech routes."""

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def audio_module(tmp_path, monkeypatch):
    servers_dir = Path(__file__).resolve().parents[1] / "servers"
    monkeypatch.syspath_prepend(str(servers_dir))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache" / "hub"))
    monkeypatch.setenv("AUDIO_PUBLIC_BASE_URL", "http://audio.test:8002")

    sys.modules.pop("audio_server", None)
    common_module = sys.modules.get("common")
    common_path = getattr(common_module, "__file__", "")
    if common_module is not None and str(Path(common_path).resolve()) != str((servers_dir / "common.py").resolve()):
        sys.modules.pop("common", None)

    module = importlib.import_module("audio_server")
    captured = {}

    def fake_runner(name, sample_rate=24000, duration=1.0):
        def run(**kwargs):
            captured[name] = kwargs
            Path(kwargs["output_path"]).write_bytes(b"WAV")
            return {"sample_rate": sample_rate, "duration_seconds": duration}

        return run

    for language in ("eng", "deu", "fra", "spa"):
        monkeypatch.setattr(
            module.mms_tts_runners[language],
            "generate",
            fake_runner(f"mms_{language}", 16000, 1.5),
        )
    monkeypatch.setattr(module.speecht5_runner, "generate", fake_runner("speecht5", 16000, 2.0))
    monkeypatch.setattr(module.f5_tts_runner, "generate", fake_runner("f5", 24000, 1.2))
    monkeypatch.setattr(module.e2_tts_runner, "generate", fake_runner("e2", 24000, 1.1))
    monkeypatch.setattr(module.kitten_tts_runner, "generate", fake_runner("kitten", 24000, 0.9))
    monkeypatch.setattr(module.vits_ljs_runner, "generate", fake_runner("vits", 22050, 1.0))
    module._captured = captured
    return module


@pytest.fixture
def client(audio_module):
    return TestClient(audio_module.app)


class TestMMSTTS:
    def test_default_parameters(self, client, audio_module):
        response = client.post("/generate/tts/mms-tts", json={"text": "Hello", "language": "eng"})
        assert response.status_code == 200
        assert audio_module._captured["mms_eng"]["speaking_rate"] == 1.0

    def test_custom_parameters(self, client, audio_module):
        response = client.post(
            "/generate/tts/mms-tts",
            json={"text": "Hallo", "language": "deu", "speaking_rate": 0.8, "noise_scale": 1.0},
        )
        assert response.status_code == 200
        assert audio_module._captured["mms_deu"]["noise_scale"] == 1.0

    def test_invalid_language_rejected(self, client):
        response = client.post("/generate/tts/mms-tts", json={"text": "test", "language": "zzz"})
        assert response.status_code == 422

    def test_rate_out_of_range_rejected(self, client):
        response = client.post("/generate/tts/mms-tts", json={"text": "test", "speaking_rate": 10.0})
        assert response.status_code == 422


class TestSpeechT5:
    def test_default_parameters(self, client, audio_module):
        response = client.post("/generate/tts/speecht5", json={"text": "Hello"})
        assert response.status_code == 200
        assert audio_module._captured["speecht5"]["speaker_index"] == 7306

    def test_custom_parameters(self, client, audio_module):
        response = client.post(
            "/generate/tts/speecht5",
            json={"text": "Hello", "speaker_index": 100, "threshold": 0.7, "maxlenratio": 30.0},
        )
        assert response.status_code == 200
        assert audio_module._captured["speecht5"]["speaker_index"] == 100

    def test_threshold_out_of_range_rejected(self, client):
        response = client.post("/generate/tts/speecht5", json={"text": "test", "threshold": 5.0})
        assert response.status_code == 422


@pytest.mark.parametrize(
    ("endpoint", "runner", "model_id"),
    [
        ("f5-tts", "f5", "f5-tts"),
        ("e2-tts", "e2", "e2-tts"),
        ("kitten-tts", "kitten", "kitten-tts"),
        ("vits-ljs", "vits", "vits-ljs"),
    ],
)
def test_active_tts_routes_return_model_metadata(client, audio_module, endpoint, runner, model_id):
    response = client.post(f"/generate/tts/{endpoint}", json={"text": "Hello"})
    assert response.status_code == 200
    assert response.json()["modelId"] == model_id


def test_f5_tts_forwards_reference_options(client, audio_module):
    response = client.post(
        "/generate/tts/f5-tts",
        json={"text": "Hello", "ref_file": "/tmp/reference.wav", "ref_text": "Reference", "speed": 1.5},
    )
    assert response.status_code == 200
    assert audio_module._captured["f5"]["speed"] == 1.5


def test_kitten_tts_forwards_voice_options(client, audio_module):
    response = client.post(
        "/generate/tts/kitten-tts",
        json={"text": "Hello", "voice": "expr-voice-1-f", "speed": 1.5},
    )
    assert response.status_code == 200
    assert audio_module._captured["kitten"]["voice"] == "expr-voice-1-f"


def test_tts_rejects_unsupported_parameters(client):
    response = client.post("/generate/tts/kitten-tts", json={"text": "test", "fake_param": "bad"})
    assert response.status_code == 422


class TestModelRegistry:
    def test_active_tts_models_are_registered(self, client):
        response = client.get("/models")
        model_ids = {model["id"] for model in response.json()["models"]}
        assert {"mms-tts", "speecht5-tts", "f5-tts", "e2-tts", "kitten-tts"}.issubset(model_ids)

    def test_mms_registry_exposes_rate_controls(self, client):
        response = client.get("/models")
        mms = next(model for model in response.json()["models"] if model["id"] == "mms-tts")
        assert mms["fields"]["speaking_rate"]["default"] == 1.0

    def test_speecht5_registry_exposes_speaker_index(self, client):
        response = client.get("/models")
        speech_t5 = next(model for model in response.json()["models"] if model["id"] == "speecht5-tts")
        assert speech_t5["fields"]["speaker_index"]["max"] == 7930
