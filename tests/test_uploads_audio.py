import importlib
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_DIR = REPO_ROOT / "model-gateway"
SERVICES_DIR = REPO_ROOT / "services"


def _import_gateway(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(GATEWAY_DIR))
    monkeypatch.syspath_prepend(str(SERVICES_DIR))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(GATEWAY_DIR / "registry.yaml"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    if "main" in sys.modules:
        del sys.modules["main"]
    return importlib.import_module("main")


def test_upload_audio_accepts_wav(monkeypatch, tmp_path):
    main = _import_gateway(monkeypatch, tmp_path)
    client = TestClient(main.app)
    resp = client.post(
        "/uploads/audio",
        files={"file": ("ref.wav", io.BytesIO(b"RIFFfake"), "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"].endswith(".wav")
    assert "/outputs/uploads/" in body["url"]
    assert Path(body["path"]).exists()
    assert Path(body["path"]).name.startswith("ref_")


def test_upload_audio_rejects_mp3(monkeypatch, tmp_path):
    main = _import_gateway(monkeypatch, tmp_path)
    client = TestClient(main.app)
    resp = client.post(
        "/uploads/audio",
        files={"file": ("ref.mp3", io.BytesIO(b"\xff\xfbfake"), "audio/mpeg")},
    )
    assert resp.status_code == 415
