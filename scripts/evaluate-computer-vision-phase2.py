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


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _average_precision(
    gt_boxes: list[tuple[str, list[float]]],
    predictions: list[tuple[str, list[float], float]],
    iou_threshold: float = 0.5,
) -> float:
    """VOC-style all-point interpolated AP for one category."""
    if not gt_boxes:
        return 0.0
    gt_by_entry: defaultdict[str, list[list[Any]]] = defaultdict(list)
    for entry_id, box in gt_boxes:
        gt_by_entry[entry_id].append([box, False])
    ordered = sorted(predictions, key=lambda p: -p[2])
    tp = [0] * len(ordered)
    fp = [0] * len(ordered)
    for i, (entry_id, box, _score) in enumerate(ordered):
        candidates = gt_by_entry.get(entry_id, [])
        best_iou, best_j = 0.0, -1
        for j, (gt_box, used) in enumerate(candidates):
            if used:
                continue
            iou = _iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1
            candidates[best_j][1] = True
        else:
            fp[i] = 1
    n_gt = len(gt_boxes)
    cum_tp = cum_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        recalls.append(cum_tp / n_gt)
        precisions.append(cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_recall = 0.0
    for recall, precision in zip(recalls, precisions):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def _mean_average_precision_50(
    entries: list[dict[str, Any]], predictions: dict[str, Any], model_key: str
) -> float:
    gt_by_cat: defaultdict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    pred_by_cat: defaultdict[str, list[tuple[str, list[float], float]]] = defaultdict(list)
    for entry in entries:
        boxes = entry.get("expected_boxes")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"{entry['id']}: expected_boxes are required for detection entries")
        for box in boxes:
            gt_by_cat[str(box["category"]).strip().lower()].append((entry["id"], box["bbox"]))
        model_predictions = predictions.get(entry["id"], {}).get(model_key)
        if model_predictions is None:
            raise ValueError(f"{entry['id']}: missing '{model_key}' predictions")
        for box in model_predictions.get("boxes", []):
            pred_by_cat[str(box["category"]).strip().lower()].append(
                (entry["id"], box["bbox"], float(box.get("score", 1.0)))
            )
    categories = sorted(gt_by_cat)
    if not categories:
        raise ValueError(f"no ground-truth categories found for {model_key}")
    ap_values = [
        _average_precision(gt_by_cat[category], pred_by_cat.get(category, []))
        for category in categories
    ]
    return sum(ap_values) / len(ap_values)


def _recall_50(entries: list[dict[str, Any]], predictions: dict[str, Any], model_key: str) -> float:
    total_gt = 0
    matched_gt = 0
    for entry in entries:
        boxes = entry.get("expected_boxes")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"{entry['id']}: expected_boxes are required for detection entries")
        gts = [[str(box["category"]).strip().lower(), box["bbox"], False] for box in boxes]
        total_gt += len(gts)
        model_predictions = predictions.get(entry["id"], {}).get(model_key)
        if model_predictions is None:
            raise ValueError(f"{entry['id']}: missing '{model_key}' predictions")
        for box in model_predictions.get("boxes", []):
            pred_category = str(box["category"]).strip().lower()
            best_iou, best_j = 0.0, -1
            for j, (gt_category, gt_box, used) in enumerate(gts):
                if used or gt_category != pred_category:
                    continue
                iou = _iou(box["bbox"], gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                gts[best_j][2] = True
        matched_gt += sum(1 for _, _, used in gts if used)
    if total_gt == 0:
        raise ValueError(f"no ground-truth boxes found for {model_key}")
    return matched_gt / total_gt


def evaluate_detection(entries: list[dict[str, Any]], predictions: dict[str, Any]) -> dict[str, float]:
    known_class = [e for e in entries if "detection_known_class" in e.get("groups", [])]
    open_vocab = [e for e in entries if "detection_open_vocabulary" in e.get("groups", [])]
    summary: dict[str, float] = {}
    if known_class:
        summary["rtdetr_map50"] = _mean_average_precision_50(known_class, predictions, "rtdetr")
        summary["yolox_map50"] = _mean_average_precision_50(known_class, predictions, "yolox")
    if open_vocab:
        summary["grounding_dino_recall50"] = _recall_50(open_vocab, predictions, "grounding_dino")
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
    predictions = load_predictions(args.predictions)
    summary: dict[str, float] = {}
    if args.tasks in ("ocr", "all"):
        summary.update(evaluate_ocr(entries, predictions))
    if args.tasks in ("detection", "all"):
        summary.update(evaluate_detection(entries, predictions))
    assert_quality(summary, args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump({"summary": summary}, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
