from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2_evaluation", ROOT / "scripts" / "evaluate-computer-vision-phase2.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_quality_gate_rejects_missing_metrics():
    with pytest.raises(SystemExit, match="quality metrics missing"):
        MODULE.assert_quality({}, "ocr")


def test_quality_gate_rejects_regression():
    summary = {metric: 0.0 for metric in MODULE.TASK_METRICS["ocr"]}
    summary["ocr_en_cer"] = 0.11
    with pytest.raises(SystemExit, match="quality gate failed"):
        MODULE.assert_quality(summary, "ocr")


def test_quality_gate_accepts_ocr_floors():
    summary = {metric: 0.0 for metric in MODULE.TASK_METRICS["ocr"]}
    MODULE.assert_quality(summary, "ocr")


def test_error_rate_calculates_character_and_word_distance():
    assert MODULE.error_rate("hello", "hallo", words=False) == pytest.approx(0.2)
    assert MODULE.error_rate("hello world", "hello", words=True) == pytest.approx(0.5)


def test_iou_of_identical_boxes_is_one():
    assert MODULE._iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert MODULE._iou([0, 0, 10, 10], [20, 20, 10, 10]) == pytest.approx(0.0)


def test_iou_of_half_overlapping_boxes():
    # [0,0,10,10] and [5,0,10,10] overlap in a 5x10 region; union is 150.
    assert MODULE._iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)


def _detection_entry(entry_id: str, category: str, bbox: list[float]) -> dict:
    return {
        "id": entry_id,
        "groups": ["detection_known_class"],
        "expected_boxes": [{"category": category, "bbox": bbox}],
    }


def test_mean_average_precision_is_one_for_a_perfect_prediction():
    entries = [_detection_entry("img-1", "cat", [0, 0, 10, 10])]
    predictions = {
        "img-1": {"rtdetr": {"boxes": [{"category": "cat", "bbox": [0, 0, 10, 10], "score": 0.9}]}}
    }
    assert MODULE._mean_average_precision_50(entries, predictions, "rtdetr") == pytest.approx(1.0)


def test_mean_average_precision_is_zero_when_nothing_is_detected():
    entries = [_detection_entry("img-1", "cat", [0, 0, 10, 10])]
    predictions = {"img-1": {"rtdetr": {"boxes": []}}}
    assert MODULE._mean_average_precision_50(entries, predictions, "rtdetr") == pytest.approx(0.0)


def test_mean_average_precision_penalizes_a_false_positive_ahead_of_the_true_positive():
    entries = [_detection_entry("img-1", "cat", [0, 0, 10, 10])]
    predictions = {
        "img-1": {
            "rtdetr": {
                "boxes": [
                    {"category": "cat", "bbox": [50, 50, 10, 10], "score": 0.95},  # false positive, ranked first
                    {"category": "cat", "bbox": [0, 0, 10, 10], "score": 0.5},  # true positive
                ]
            }
        }
    }
    ap = MODULE._mean_average_precision_50(entries, predictions, "rtdetr")
    assert 0.0 < ap < 1.0


def test_mean_average_precision_raises_when_predictions_are_missing_for_the_model():
    entries = [_detection_entry("img-1", "cat", [0, 0, 10, 10])]
    with pytest.raises(ValueError, match="missing 'rtdetr' predictions"):
        MODULE._mean_average_precision_50(entries, {}, "rtdetr")


def test_recall_50_counts_only_matching_category_and_iou():
    entries = [
        {
            "id": "img-1",
            "groups": ["detection_open_vocabulary"],
            "expected_boxes": [
                {"category": "gizmo", "bbox": [0, 0, 10, 10]},
                {"category": "widget", "bbox": [100, 100, 10, 10]},
            ],
        }
    ]
    predictions = {
        "img-1": {
            "grounding_dino": {
                "boxes": [
                    {"category": "gizmo", "bbox": [0, 0, 10, 10]},  # matches
                    {"category": "widget", "bbox": [0, 0, 1, 1]},  # wrong location, no match
                ]
            }
        }
    }
    assert MODULE._recall_50(entries, predictions, "grounding_dino") == pytest.approx(0.5)


def test_evaluate_detection_routes_known_class_and_open_vocab_separately():
    entries = [
        _detection_entry("img-1", "cat", [0, 0, 10, 10]),
        {
            "id": "img-2",
            "groups": ["detection_open_vocabulary"],
            "expected_boxes": [{"category": "gizmo", "bbox": [0, 0, 10, 10]}],
        },
    ]
    predictions = {
        "img-1": {
            "rtdetr": {"boxes": [{"category": "cat", "bbox": [0, 0, 10, 10], "score": 0.9}]},
            "yolox": {"boxes": [{"category": "cat", "bbox": [0, 0, 10, 10], "score": 0.9}]},
        },
        "img-2": {"grounding_dino": {"boxes": [{"category": "gizmo", "bbox": [0, 0, 10, 10]}]}},
    }
    summary = MODULE.evaluate_detection(entries, predictions)
    assert summary == {
        "rtdetr_map50": pytest.approx(1.0),
        "yolox_map50": pytest.approx(1.0),
        "grounding_dino_recall50": pytest.approx(1.0),
    }
