#!/usr/bin/env python3
"""Verify the frozen evaluation corpus and apply Phase 2 quality floors."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


QUALITY_FLOORS = {
    "ocr_en_cer": ("max", 0.10),
    "ocr_en_wer": ("max", 0.20),
    "ocr_fi_cer": ("max", 0.10),
    "ocr_fi_wer": ("max", 0.20),
    "ocr_sv_cer": ("max", 0.10),
    "ocr_sv_wer": ("max", 0.20),
    "ocr_mixed_cer": ("max", 0.10),
    "ocr_mixed_wer": ("max", 0.20),
    "rtdetr_map50": ("min", 0.45),
    "yolox_map50": ("min", 0.35),
    "grounding_dino_recall50": ("min", 0.60),
}

TASK_METRICS = {
    "ocr": {metric for metric in QUALITY_FLOORS if metric.startswith("ocr_")},
    "detection": {
        "rtdetr_map50",
        "yolox_map50",
        "grounding_dino_recall50",
    },
}
TASK_METRICS["all"] = TASK_METRICS["ocr"] | TASK_METRICS["detection"]


def assert_quality(summary: dict[str, float], task_group: str) -> None:
    if task_group not in TASK_METRICS:
        raise ValueError(f"unknown task group: {task_group}")
    required = TASK_METRICS[task_group]
    missing = required - summary.keys()
    if missing:
        raise SystemExit("quality metrics missing: " + ", ".join(sorted(missing)))
    failures = [
        metric
        for metric, (direction, floor) in QUALITY_FLOORS.items()
        if metric in required
        and (
            (direction == "max" and summary[metric] > floor)
            or (direction == "min" and summary[metric] < floor)
        )
    ]
    if failures:
        raise SystemExit("quality gate failed: " + ", ".join(sorted(failures)))


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def error_rate(expected: str, actual: str, *, words: bool) -> float:
    expected_tokens = expected.split() if words else list(expected)
    actual_tokens = actual.split() if words else list(actual)
    if not expected_tokens:
        return 0.0 if not actual_tokens else 1.0
    return _edit_distance(expected_tokens, actual_tokens) / len(expected_tokens)


def verify_manifest(manifest: dict[str, Any], corpus_root: Path) -> list[dict[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest entries are required")
    minimum_counts = manifest.get("minimum_counts", {})
    if not isinstance(minimum_counts, dict):
        raise ValueError("minimum_counts must be a mapping")
    groups: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every manifest entry must be a mapping")
        required = ("id", "host_path", "sha256", "license", "provenance", "split", "media_type", "safety_review")
        missing = [field for field in required if not entry.get(field)]
        if missing:
            raise ValueError(f"{entry.get('id', '<unknown>')}: missing {', '.join(missing)}")
        path = (corpus_root / entry["host_path"]).resolve()
        if corpus_root.resolve() not in path.parents:
            raise ValueError(f"{entry['id']}: host_path escapes corpus root")
        if not path.is_file():
            raise ValueError(f"{entry['id']}: fixture is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['id']}: fixture checksum mismatch")
        for group in entry.get("groups", []):
            groups[str(group)] += 1
    for group, minimum in minimum_counts.items():
        if not isinstance(minimum, int) or minimum < 0:
            raise ValueError(f"{group}: minimum count must be a non-negative integer")
        if groups[group] < minimum:
            raise ValueError(f"{group}: expected {minimum} fixtures, found {groups[group]}")
    return entries


def evaluate_ocr(entries: list[dict[str, Any]], predictions: dict[str, Any]) -> dict[str, float]:
    grouped: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for entry in entries:
        expected = entry.get("expected_text")
        prediction = predictions.get(entry["id"], {}).get("text")
        if not isinstance(expected, str) or not isinstance(prediction, str):
            raise ValueError(f"{entry['id']}: OCR expected_text and prediction are required")
        language = str(entry.get("language", "mixed")).lower()
        grouped[language].append((expected, prediction))
    summary: dict[str, float] = {}
    for language, values in grouped.items():
        key = language if language in {"en", "fi", "sv"} else "mixed"
        summary[f"ocr_{key}_cer"] = sum(error_rate(expected, actual, words=False) for expected, actual in values) / len(values)
        summary[f"ocr_{key}_wer"] = sum(error_rate(expected, actual, words=True) for expected, actual in values) / len(values)
    return summary


def load_predictions(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("predictions must be a mapping keyed by fixture id")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", choices=("ocr", "detection", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a mapping")
    corpus_root = args.corpus_root or args.manifest.parent
    entries = verify_manifest(manifest, corpus_root)
    if args.predictions is None:
        raise SystemExit("authenticated predictions are required for the quality gate")
    summary = evaluate_ocr(entries, load_predictions(args.predictions)) if args.tasks == "ocr" else {}
    assert_quality(summary, args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump({"summary": summary}, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
