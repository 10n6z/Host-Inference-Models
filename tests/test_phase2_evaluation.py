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
