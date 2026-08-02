import base64
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "services" / "vision-grounding-dino"


@pytest.fixture
def service_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SERVICE_DIR))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "services" / "vision-common"))
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


class _FakeTensor(list):
    def tolist(self):
        return list(self)


class _FakeProcessor:
    def __init__(self, result):
        self._result = result
        self.seen_labels = None

    def __call__(self, text, images, return_tensors):
        self.seen_labels = text[0]
        return _FakeBatch()

    def post_process_object_detection(self, outputs, target_sizes, threshold):
        return [self._result]


class _FakeBatch:
    def to(self, device):
        return self

    def keys(self):
        return []

    def __getitem__(self, key):
        raise KeyError(key)


class _FakeModel:
    def __call__(self, **kwargs):
        return object()

    def eval(self):
        return self

    def to(self, device):
        return self


def _install_fake_model(service_module, monkeypatch, result):
    fake_processor = _FakeProcessor(result)
    fake_model = _FakeModel()

    class _FakeTorch:
        @staticmethod
        def inference_mode():
            import contextlib

            return contextlib.nullcontext()

        @staticmethod
        def tensor(value):
            return value

        class cuda:
            @staticmethod
            def is_available():
                return False

    service_module._model_state.update(
        {
            "processor": fake_processor,
            "model": fake_model,
            "device": "cpu",
            "torch": _FakeTorch,
        }
    )
    return fake_processor


def test_health_reports_service_name(client):
    response = client.get("/health")

    assert response.json()["service"] == "vision-grounding-dino"


def test_detect_rejects_invalid_base64(client):
    response = client.post(
        "/generate/detect/grounding-dino", json={"image": "not-base64!!"}
    )

    assert response.status_code == 400


def test_detect_requires_labels_or_prompt(client):
    response = client.post(
        "/generate/detect/grounding-dino", json={"image": _png_b64()}
    )

    assert response.status_code == 400
    assert "labels or prompt" in response.json()["message"]


def test_detect_builds_lowercased_label_list_from_labels(client, service_module, monkeypatch):
    fake_processor = _install_fake_model(
        service_module,
        monkeypatch,
        {"scores": _FakeTensor([0.81]), "labels": _FakeTensor([0]), "boxes": _FakeTensor([[10.0, 10.0, 60.0, 40.0]])},
    )

    response = client.post(
        "/generate/detect/grounding-dino",
        json={"image": _png_b64(100, 100), "labels": ["Forklift", "Ladder"]},
    )

    assert response.status_code == 200
    assert fake_processor.seen_labels == ["forklift", "ladder"]


def test_detect_returns_normalized_detections(client, service_module, monkeypatch):
    _install_fake_model(
        service_module,
        monkeypatch,
        {
            "scores": _FakeTensor([0.9]),
            "labels": _FakeTensor([0]),
            "boxes": _FakeTensor([[10.0, 10.0, 60.0, 40.0]]),
        },
    )

    response = client.post(
        "/generate/detect/grounding-dino",
        json={"image": _png_b64(100, 100), "prompt": "a forklift"},
    )

    assert response.status_code == 200
    detections = response.json()["detections"]
    assert detections == [
        {
            "label": "a forklift",
            "confidence": 0.9,
            "box": {"x": 10.0, "y": 10.0, "width": 50.0, "height": 30.0},
        }
    ]


def test_detect_drops_zero_area_boxes(client, service_module, monkeypatch):
    _install_fake_model(
        service_module,
        monkeypatch,
        {
            "scores": _FakeTensor([0.9]),
            "labels": _FakeTensor([0]),
            "boxes": _FakeTensor([[10.0, 10.0, 10.0, 40.0]]),
        },
    )

    response = client.post(
        "/generate/detect/grounding-dino",
        json={"image": _png_b64(100, 100), "prompt": "forklift"},
    )

    assert response.status_code == 200
    assert response.json()["detections"] == []
