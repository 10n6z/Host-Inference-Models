import base64
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "services" / "vision-tesseract"


@pytest.fixture
def service_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SERVICE_DIR))
    if "main" in sys.modules:
        del sys.modules["main"]
    import main

    return main


@pytest.fixture
def client(service_module):
    return TestClient(service_module.app)


def _png_b64(width=40, height=20):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_health_reports_service_name(client):
    response = client.get("/health")

    assert response.json()["service"] == "vision-tesseract"


def test_ocr_rejects_invalid_base64(client):
    response = client.post(
        "/generate/ocr/tesseract", json={"image": "not-base64!!", "language": "en"}
    )

    assert response.status_code == 400


def test_ocr_rejects_oversized_declared_dimensions(client, monkeypatch, service_module):
    monkeypatch.setattr(service_module, "MAX_IMAGE_PIXELS", 100)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    response = client.post(
        "/generate/ocr/tesseract", json={"image": _png_b64(200, 200), "language": "en"}
    )

    assert response.status_code == 413


def test_ocr_returns_normalized_words(client, monkeypatch, service_module):
    fake_data = {
        "text": ["", "Hello", "world", "---"],
        "conf": [-1, 92.5, 87.0, -1],
        "left": [0, 10, 60, 0],
        "top": [0, 5, 5, 0],
        "width": [0, 40, 35, 0],
        "height": [0, 12, 12, 0],
    }
    monkeypatch.setattr(
        service_module.pytesseract, "image_to_data", lambda *a, **k: fake_data
    )

    response = client.post(
        "/generate/ocr/tesseract", json={"image": _png_b64(), "language": "en"}
    )

    assert response.status_code == 200
    words = response.json()["words"]
    assert [w["text"] for w in words] == ["Hello", "world"]
    assert words[0]["confidence"] == pytest.approx(0.925)
    assert words[0]["box"] == {"x": 10.0, "y": 5.0, "width": 40.0, "height": 12.0}


def test_ocr_drops_low_confidence_boundary_entries(monkeypatch, service_module, client):
    # conf == -1 marks a block/line/paragraph boundary in pytesseract's DICT
    # output, not a real recognized token, and must never surface as a word.
    fake_data = {
        "text": ["Header"],
        "conf": [-1],
        "left": [0],
        "top": [0],
        "width": [50],
        "height": [10],
    }
    monkeypatch.setattr(
        service_module.pytesseract, "image_to_data", lambda *a, **k: fake_data
    )

    response = client.post(
        "/generate/ocr/tesseract", json={"image": _png_b64(), "language": "en"}
    )

    assert response.json()["words"] == []


def test_language_selection_maps_to_pinned_tesseract_codes(service_module):
    assert service_module.LANGUAGE_MAP["en"] == "eng"
    assert service_module.LANGUAGE_MAP["fi"] == "fin"
    assert service_module.LANGUAGE_MAP["sv"] == "swe"
    assert service_module.LANGUAGE_MAP["auto"] == "eng+fin+swe"
