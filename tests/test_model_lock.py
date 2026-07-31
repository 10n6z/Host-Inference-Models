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


def valid_host_deployment_report(**overrides):
    report = {
        "model_id": "vision-planner",
        "resolved_revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "deployment": "host",
        "environment_digest": "2e85c3774cb99426024dad112f17d263bac1831ba859053eff17014fa74d6ca4",
        "license": "apache-2.0",
        "cache_bytes": 14496078512,
        "package_pins": {"vllm": "0.26.0", "torch": "2.11.0"},
    }
    report.update(overrides)
    return report


def test_model_lock_accepts_a_host_deployment_with_an_environment_digest():
    lock = MODULE.build_model_lock([valid_host_deployment_report()])
    entry = lock["models"]["vision-planner"]
    assert entry["deployment"] == "host"
    assert entry["environment_digest"] == (
        "2e85c3774cb99426024dad112f17d263bac1831ba859053eff17014fa74d6ca4"
    )
    assert "container_digest" not in entry


def test_model_lock_rejects_a_host_deployment_missing_environment_digest():
    report = valid_host_deployment_report()
    del report["environment_digest"]
    with pytest.raises(ValueError, match="environment_digest"):
        MODULE.build_model_lock([report])


def test_model_lock_rejects_a_non_hex_environment_digest():
    with pytest.raises(ValueError, match="environment_digest"):
        MODULE.build_model_lock(
            [valid_host_deployment_report(environment_digest="not-a-real-digest")]
        )


def test_model_lock_rejects_an_unknown_deployment_kind():
    with pytest.raises(ValueError, match="deployment"):
        MODULE.build_model_lock([valid_host_deployment_report(deployment="serverless")])


def test_model_lock_defaults_to_container_deployment_for_backward_compatibility():
    lock = MODULE.build_model_lock([valid_preflight_report()])
    assert lock["models"]["pp-ocr-v4"]["deployment"] == "container"
