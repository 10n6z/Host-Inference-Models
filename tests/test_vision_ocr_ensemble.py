import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "services" / "vision-ocr-ensemble"


@pytest.fixture
def service_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SERVICE_DIR))
    if "main" in sys.modules:
        del sys.modules["main"]
    import main

    return main


def _word(module, text, confidence, box, review_required=False):
    return module.OcrWord(
        text=text,
        confidence=confidence,
        box=module.WordBox(x=box[0], y=box[1], width=box[2], height=box[3]),
        review_required=review_required,
    )


def test_ensemble_marks_disagreement_for_review(service_module):
    result = service_module.combine_words(
        paddle=[_word(service_module, "kypärä", 0.91, (10, 10, 80, 20))],
        tesseract=[_word(service_module, "kypara", 0.74, (11, 10, 79, 20))],
    )

    assert result["words"][0]["text"] == "kypärä"
    assert result["words"][0]["review_required"] is True


def test_ensemble_does_not_flag_agreement(service_module):
    result = service_module.combine_words(
        paddle=[_word(service_module, "hello", 0.91, (10, 10, 80, 20))],
        tesseract=[_word(service_module, "Hello", 0.74, (11, 10, 79, 20))],
    )

    assert result["words"][0]["review_required"] is False


def test_ensemble_keeps_unmatched_words_from_both_engines(service_module):
    result = service_module.combine_words(
        paddle=[_word(service_module, "left", 0.9, (0, 0, 20, 10))],
        tesseract=[_word(service_module, "right", 0.9, (500, 500, 20, 10))],
    )

    texts = {word["text"] for word in result["words"]}
    assert texts == {"left", "right"}
    assert all(word["review_required"] is False for word in result["words"])


def test_ensemble_prefers_higher_confidence_engine(service_module):
    result = service_module.combine_words(
        paddle=[_word(service_module, "cat", 0.99, (0, 0, 30, 10))],
        tesseract=[_word(service_module, "cot", 0.40, (0, 0, 30, 10))],
    )

    assert result["words"][0]["text"] == "cat"


def test_box_iou_below_threshold_does_not_match(service_module):
    result = service_module.combine_words(
        paddle=[_word(service_module, "a", 0.9, (0, 0, 10, 10))],
        tesseract=[_word(service_module, "b", 0.9, (100, 100, 10, 10))],
    )

    assert len(result["words"]) == 2


def test_ocr_returns_502_when_both_engines_unavailable(service_module, monkeypatch):
    async def _empty(*args, **kwargs):
        return []

    monkeypatch.setattr(service_module, "_call_engine", _empty)
    client = TestClient(service_module.app)

    response = client.post("/generate/ocr/ensemble", json={"image": "x", "language": "en"})

    assert response.status_code == 502
