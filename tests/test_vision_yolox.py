import base64
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "services" / "vision-yolox"


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


class _FakeResultTensor(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    def __call__(self, batch):
        return "raw-output"


class _FakeTorch:
    @staticmethod
    def from_numpy(array):
        class _Tensor:
            def unsqueeze(self, dim):
                return self

            def to(self, device):
                return self

        return _Tensor()

    @staticmethod
    def inference_mode():
        import contextlib

        return contextlib.nullcontext()

    class cuda:
        @staticmethod
        def is_available():
            return False


def _install_fake_model(service_module, postprocess_result):
    service_module._model_state.update(
        {
            "model": _FakeModel(),
            "device": "cpu",
            "torch": _FakeTorch,
            "postprocess": lambda *a, **k: postprocess_result,
        }
    )


def test_health_reports_service_name(client):
    response = client.get("/health")

    assert response.json()["service"] == "vision-yolox"


def test_detect_rejects_invalid_base64(client):
    response = client.post("/generate/detect/yolox", json={"image": "not-base64!!"})

    assert response.status_code == 400


def test_detect_returns_normalized_detections(client, service_module):
    # x0,y0,x1,y1,obj_conf,class_conf,class_idx -- person is COCO_CLASSES[0]
    _install_fake_model(
        service_module,
        [_FakeResultTensor([[10.0, 10.0, 60.0, 40.0, 0.9, 0.9, 0]])],
    )

    response = client.post(
        "/generate/detect/yolox",
        json={"image": _png_b64(100, 100)},
    )

    assert response.status_code == 200
    detections = response.json()["detections"]
    assert len(detections) == 1
    assert detections[0]["label"] == "person"
    assert detections[0]["confidence"] == pytest.approx(0.81)


def test_detect_returns_empty_list_when_nothing_passes_threshold(client, service_module):
    _install_fake_model(service_module, [None])

    response = client.post(
        "/generate/detect/yolox",
        json={"image": _png_b64(100, 100)},
    )

    assert response.status_code == 200
    assert response.json()["detections"] == []


def test_detect_drops_out_of_range_class_index(client, service_module):
    _install_fake_model(
        service_module,
        [_FakeResultTensor([[10.0, 10.0, 60.0, 40.0, 0.9, 0.9, 999]])],
    )

    response = client.post(
        "/generate/detect/yolox",
        json={"image": _png_b64(100, 100)},
    )

    assert response.status_code == 200
    assert response.json()["detections"] == []
