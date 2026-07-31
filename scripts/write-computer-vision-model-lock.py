#!/usr/bin/env python3
"""Build a stable lock from immutable Computer Vision preflight reports."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


REVISION = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)


def require_revision(report: dict[str, Any]) -> str:
    revision = str(report.get("resolved_revision", ""))
    if not REVISION.fullmatch(revision):
        raise ValueError("preflight did not resolve an immutable revision")
    return revision


def _require_report_field(report: dict[str, Any], field: str) -> str:
    value = report.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"preflight report requires {field}")
    return value.strip()


def _require_immutability_anchor(report: dict[str, Any], model_id: str) -> dict[str, str]:
    """Every entry needs *some* immutable anchor a rebuild can be verified
    against. Container deployments anchor on the built image digest;
    host-process deployments (e.g. vLLM services -- see
    docs/computer-vision/model-preflight-2026-07-31.md for why some models
    cannot run containerized on this host) anchor on the sha256 of the
    exact resolved dependency closure ('pip freeze'), which is the
    equivalent integrity guarantee for a process that has no image to
    hash."""
    deployment = report.get("deployment", "container")
    if deployment not in ("container", "host"):
        raise ValueError(f"{model_id}: deployment must be 'container' or 'host'")
    if deployment == "container":
        digest = _require_report_field(report, "container_digest")
        if not DIGEST.fullmatch(digest):
            raise ValueError(f"{model_id}: container_digest must be immutable")
        return {"deployment": "container", "container_digest": digest}
    digest = _require_report_field(report, "environment_digest")
    if not re.fullmatch(r"[0-9a-f]{64}", digest, re.IGNORECASE):
        raise ValueError(f"{model_id}: environment_digest must be a sha256 hex digest")
    return {"deployment": "host", "environment_digest": digest}


def build_model_lock(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one preflight report is required")
    models: dict[str, dict[str, Any]] = {}
    for report in reports:
        model_id = _require_report_field(report, "model_id")
        if model_id in models:
            raise ValueError(f"duplicate model_id: {model_id}")
        anchor = _require_immutability_anchor(report, model_id)
        license_name = _require_report_field(report, "license")
        cache_bytes = report.get("cache_bytes")
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes < 0:
            raise ValueError(f"{model_id}: cache_bytes must be a non-negative integer")
        package_pins = report.get("package_pins", {})
        if not isinstance(package_pins, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in package_pins.items()
        ):
            raise ValueError(f"{model_id}: package_pins must be a string mapping")
        models[model_id] = {
            "revision": require_revision(report),
            **anchor,
            "license": license_name,
            "cache_bytes": cache_bytes,
            "package_pins": dict(sorted(package_pins.items())),
        }
    return {"version": "computer-vision-model-lock-v1", "models": dict(sorted(models.items()))}


def load_reports(directory: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path}: report must be an object")
        reports.append(value)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = build_model_lock(load_reports(args.preflight_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
