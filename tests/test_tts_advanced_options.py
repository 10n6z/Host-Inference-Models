"""
Test all 7 working TTS models with advanced configuration options.

Models tested:
  1. MMS-TTS eng
  2. MMS-TTS deu
  3. MMS-TTS fra
  4. SpeechT5
  5. Bark Small
  6. Kokoro-82M (native)
  7. Kokoro-82M ONNX
"""

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

    if "audio_server" in sys.modules:
        del sys.modules["audio_server"]

    module = importlib.import_module("audio_server")

    captured = {}

    def _fake_mms(lang):
        def _run(**kwargs):
            captured[f"mms_{lang}"] = kwargs
            Path(kwargs["output_path"]).write_bytes(b"WAV")
            return {"sample_rate": 16000, "duration_seconds": 1.5}
        return _run

    def _fake_speecht5(**kwargs):
        captured["speecht5"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 16000, "duration_seconds": 2.0}

    def _fake_bark(**kwargs):
        captured["bark"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 24000, "duration_seconds": 3.0}

    def _fake_kokoro(**kwargs):
        captured["kokoro"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 24000, "duration_seconds": 1.2}

    def _fake_kokoro_onnx(**kwargs):
        captured["kokoro_onnx"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"WAV")
        return {"sample_rate": 24000, "duration_seconds": 1.0}

    for lang in ("eng", "deu", "fra"):
        monkeypatch.setattr(module.mms_tts_runners[lang], "generate", _fake_mms(lang))
    monkeypatch.setattr(module.speecht5_runner, "generate", _fake_speecht5)
    monkeypatch.setattr(module.bark_runner, "generate", _fake_bark)
    monkeypatch.setattr(module.kokoro_runner, "generate", _fake_kokoro)
    monkeypatch.setattr(module.kokoro_onnx_runner, "generate", _fake_kokoro_onnx)

    module._captured = captured
    return module


@pytest.fixture
def client(audio_module):
    return TestClient(audio_module.app)


# ─── MMS-TTS Tests ───────────────────────────────────────────────────────────


class TestMMSTTS:
    def test_eng_default_params(self, client, audio_module):
        res = client.post("/generate/tts/mms-tts", json={"text": "Hello world", "language": "eng"})
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["modelId"] == "mms-tts-eng"
        params = audio_module._captured["mms_eng"]
        assert params["speaking_rate"] == 1.0
        assert params["noise_scale"] == 0.667
        assert params["noise_scale_duration"] == 0.8

    def test_eng_custom_speaking_rate(self, client, audio_module):
        res = client.post("/generate/tts/mms-tts", json={
            "text": "Fast speech test",
            "language": "eng",
            "speaking_rate": 2.0,
            "noise_scale": 0.4,
            "noise_scale_duration": 0.5,
        })
        assert res.status_code == 200
        params = audio_module._captured["mms_eng"]
        assert params["speaking_rate"] == 2.0
        assert params["noise_scale"] == 0.4
        assert params["noise_scale_duration"] == 0.5

    def test_deu_with_all_options(self, client, audio_module):
        res = client.post("/generate/tts/mms-tts", json={
            "text": "Hallo Welt",
            "language": "deu",
            "speaking_rate": 0.8,
            "noise_scale": 1.0,
            "noise_scale_duration": 1.2,
        })
        assert res.status_code == 200
        body = res.json()
        assert body["modelId"] == "mms-tts-deu"
        params = audio_module._captured["mms_deu"]
        assert params["speaking_rate"] == 0.8
        assert params["noise_scale"] == 1.0
        assert params["noise_scale_duration"] == 1.2

    def test_fra_with_all_options(self, client, audio_module):
        res = client.post("/generate/tts/mms-tts", json={
            "text": "Bonjour le monde",
            "language": "fra",
            "speaking_rate": 1.5,
            "noise_scale": 0.3,
            "noise_scale_duration": 0.6,
        })
        assert res.status_code == 200
        body = res.json()
        assert body["modelId"] == "mms-tts-fra"
        params = audio_module._captured["mms_fra"]
        assert params["speaking_rate"] == 1.5

    def test_invalid_language_rejected(self, client):
        res = client.post("/generate/tts/mms-tts", json={"text": "test", "language": "zzz"})
        assert res.status_code == 422

    def test_speaking_rate_out_of_range_rejected(self, client):
        res = client.post("/generate/tts/mms-tts", json={
            "text": "test",
            "language": "eng",
            "speaking_rate": 10.0,
        })
        assert res.status_code == 422

    def test_noise_scale_out_of_range_rejected(self, client):
        res = client.post("/generate/tts/mms-tts", json={
            "text": "test",
            "language": "eng",
            "noise_scale": 5.0,
        })
        assert res.status_code == 422


# ─── SpeechT5 Tests ──────────────────────────────────────────────────────────


class TestSpeechT5:
    def test_default_params(self, client, audio_module):
        res = client.post("/generate/tts/speecht5", json={"text": "Hello from SpeechT5"})
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["modelId"] == "speecht5-tts"
        params = audio_module._captured["speecht5"]
        assert params["speaker_index"] == 7306
        assert params["threshold"] == 0.5
        assert params["minlenratio"] == 0.0
        assert params["maxlenratio"] == 20.0

    def test_custom_speaker_and_ratios(self, client, audio_module):
        res = client.post("/generate/tts/speecht5", json={
            "text": "Custom speaker test",
            "speaker_index": 100,
            "threshold": 0.7,
            "minlenratio": 0.5,
            "maxlenratio": 30.0,
        })
        assert res.status_code == 200
        params = audio_module._captured["speecht5"]
        assert params["speaker_index"] == 100
        assert params["threshold"] == 0.7
        assert params["minlenratio"] == 0.5
        assert params["maxlenratio"] == 30.0

    def test_speaker_index_out_of_range_rejected(self, client):
        res = client.post("/generate/tts/speecht5", json={
            "text": "test",
            "speaker_index": 99999,
        })
        assert res.status_code == 422

    def test_threshold_out_of_range_rejected(self, client):
        res = client.post("/generate/tts/speecht5", json={
            "text": "test",
            "threshold": 5.0,
        })
        assert res.status_code == 422

    def test_maxlenratio_below_min_rejected(self, client):
        res = client.post("/generate/tts/speecht5", json={
            "text": "test",
            "maxlenratio": 0.5,
        })
        assert res.status_code == 422


# ─── Bark Tests ───────────────────────────────────────────────────────────────


class TestBark:
    def test_default_params(self, client, audio_module):
        res = client.post("/generate/tts/bark", json={"text": "Hello from Bark"})
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["modelId"] == "bark-small"
        params = audio_module._captured["bark"]
        assert params["voice_preset"] == "v2/en_speaker_6"
        assert params["do_sample"] is True
        assert params["temperature"] == 1.0
        assert params["semantic_temperature"] == 1.0
        assert params["coarse_temperature"] == 1.0
        assert params["fine_temperature"] == 1.0

    def test_custom_voice_and_temperatures(self, client, audio_module):
        res = client.post("/generate/tts/bark", json={
            "text": "Custom temperature test",
            "voice_preset": "v2/en_speaker_3",
            "do_sample": True,
            "temperature": 0.7,
            "semantic_temperature": 0.8,
            "coarse_temperature": 0.6,
            "fine_temperature": 0.9,
            "semantic_max_new_tokens": 512,
            "coarse_max_new_tokens": 1024,
            "fine_max_new_tokens": 1024,
        })
        assert res.status_code == 200
        params = audio_module._captured["bark"]
        assert params["voice_preset"] == "v2/en_speaker_3"
        assert params["temperature"] == 0.7
        assert params["semantic_temperature"] == 0.8
        assert params["coarse_temperature"] == 0.6
        assert params["fine_temperature"] == 0.9
        assert params["semantic_max_new_tokens"] == 512
        assert params["coarse_max_new_tokens"] == 1024
        assert params["fine_max_new_tokens"] == 1024

    def test_deterministic_no_sample(self, client, audio_module):
        res = client.post("/generate/tts/bark", json={
            "text": "Deterministic generation",
            "do_sample": False,
            "temperature": 0.0,
            "semantic_temperature": 0.0,
            "coarse_temperature": 0.0,
            "fine_temperature": 0.0,
        })
        assert res.status_code == 200
        params = audio_module._captured["bark"]
        assert params["do_sample"] is False
        assert params["temperature"] == 0.0

    def test_temperature_out_of_range_rejected(self, client):
        res = client.post("/generate/tts/bark", json={
            "text": "test",
            "temperature": 5.0,
        })
        assert res.status_code == 422

    def test_max_tokens_out_of_range_rejected(self, client):
        res = client.post("/generate/tts/bark", json={
            "text": "test",
            "semantic_max_new_tokens": 9999,
        })
        assert res.status_code == 422

    def test_response_includes_all_params(self, client):
        res = client.post("/generate/tts/bark", json={
            "text": "Params in response",
            "voice_preset": "v2/en_speaker_9",
            "semantic_temperature": 0.5,
        })
        assert res.status_code == 200
        body = res.json()
        pu = body["parameters_used"]
        assert pu["voice_preset"] == "v2/en_speaker_9"
        assert pu["semantic_temperature"] == 0.5
        assert pu["do_sample"] is True


# ─── Kokoro-82M (native) Tests ───────────────────────────────────────────────


class TestKokoro:
    def test_default_params(self, client, audio_module):
        res = client.post("/generate/tts/kokoro", json={"text": "Hello Kokoro", "language": "en"})
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        params = audio_module._captured["kokoro"]
        assert params["voice"] == "af_heart"
        assert params["speed"] == 1.0

    def test_custom_voice_speed_language(self, client, audio_module):
        res = client.post("/generate/tts/kokoro", json={
            "text": "Custom kokoro",
            "voice": "af_bella",
            "language": "en-gb",
            "speed": 1.5,
        })
        assert res.status_code == 200
        params = audio_module._captured["kokoro"]
        assert params["voice"] == "af_bella"
        assert params["speed"] == 1.5
        assert params["lang_code"] == "b"

    def test_speed_out_of_range_rejected(self, client):
        res = client.post("/generate/tts/kokoro", json={
            "text": "test",
            "language": "en",
            "speed": 5.0,
        })
        assert res.status_code == 422


# ─── Kokoro-ONNX Tests ───────────────────────────────────────────────────────


class TestKokoroONNX:
    def test_default_params(self, client, audio_module):
        res = client.post("/generate/tts/kokoro-onnx", json={"text": "Hello ONNX Kokoro"})
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["modelId"] == "kokoro-82m-onnx"
        params = audio_module._captured["kokoro_onnx"]
        assert params["voice"] == "af_heart"
        assert params["speed"] == 1.0

    def test_custom_voice_and_speed(self, client, audio_module):
        res = client.post("/generate/tts/kokoro-onnx", json={
            "text": "Fast ONNX test",
            "voice": "am_adam",
            "speed": 2.5,
        })
        assert res.status_code == 200
        params = audio_module._captured["kokoro_onnx"]
        assert params["voice"] == "am_adam"
        assert params["speed"] == 2.5

    def test_speed_out_of_range_rejected(self, client):
        res = client.post("/generate/tts/kokoro-onnx", json={
            "text": "test",
            "speed": 10.0,
        })
        assert res.status_code == 422

    def test_unsupported_param_rejected(self, client):
        res = client.post("/generate/tts/kokoro-onnx", json={
            "text": "test",
            "fake_param": "bad",
        })
        assert res.status_code == 422
        body = res.json()
        assert body["code"] == "UNSUPPORTED_PARAMETER"


# ─── Model Registry Tests ────────────────────────────────────────────────────


class TestModelRegistry:
    def test_all_7_models_in_registry(self, client):
        res = client.get("/models")
        assert res.status_code == 200
        models = res.json()["models"]
        model_ids = {m["id"] for m in models}
        assert "mms-tts" in model_ids
        assert "speecht5-tts" in model_ids
        assert "bark-small" in model_ids
        assert "kokoro-82m" in model_ids
        assert "kokoro-82m-onnx" in model_ids

    def test_mms_registry_has_all_fields(self, client):
        res = client.get("/models")
        models = res.json()["models"]
        mms = next(m for m in models if m["id"] == "mms-tts")
        assert "speaking_rate" in mms["fields"]
        assert "noise_scale" in mms["fields"]
        assert "noise_scale_duration" in mms["fields"]
        assert mms["fields"]["speaking_rate"]["default"] == 1.0

    def test_bark_registry_has_temperature_fields(self, client):
        res = client.get("/models")
        models = res.json()["models"]
        bark = next(m for m in models if m["id"] == "bark-small")
        assert "semantic_temperature" in bark["fields"]
        assert "coarse_temperature" in bark["fields"]
        assert "fine_temperature" in bark["fields"]
        assert "do_sample" in bark["fields"]

    def test_speecht5_registry_has_speaker_index(self, client):
        res = client.get("/models")
        models = res.json()["models"]
        st5 = next(m for m in models if m["id"] == "speecht5-tts")
        assert "speaker_index" in st5["fields"]
        assert st5["fields"]["speaker_index"]["max"] == 7930
