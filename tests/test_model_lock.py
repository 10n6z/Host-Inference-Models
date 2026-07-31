from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "write_model_lock", ROOT / "scripts" / "write-computer-vision-model-lock.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_preflight_report(**overrides):
    report = {
        "model_id": "pp-ocr-v4",
        "resolved_revision": "0123456789abcdef0123456789abcdef01234567",
        "container_digest": "sha256:" + "a" * 64,
        "license": "apache-2.0",
        "cache_bytes": 1024,
        "package_pins": {"paddleocr": "2.7.3"},
    }
    report.update(overrides)
    return report


def test_model_lock_rejects_mutable_revision():
    with pytest.raises(ValueError, match="immutable revision"):
        MODULE.build_model_lock([valid_preflight_report(resolved_revision="main")])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("container_digest", "latest", "container_digest"),
        ("license", "", "license"),
        ("cache_bytes", "1024", "cache_bytes"),
    ],
)
def test_model_lock_rejects_incomplete_attestation(field, value, message):
    with pytest.raises(ValueError, match=message):
        MODULE.build_model_lock([valid_preflight_report(**{field: value})])


def test_model_lock_sorts_models_and_package_pins():
    lock = MODULE.build_model_lock(
        [
            valid_preflight_report(model_id="z-model", package_pins={"z": "1", "a": "2"}),
            valid_preflight_report(model_id="a-model"),
        ]
    )
    assert list(lock["models"]) == ["a-model", "z-model"]
    assert list(lock["models"]["z-model"]["package_pins"]) == ["a", "z"]
