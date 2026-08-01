#!/usr/bin/env python3
"""Call the real vision containers directly (no gateway/JWT -- per the
2026-08-01 decision to skip authenticated-gateway auth for this eval run,
documented in docs/superpowers/plans/2026-07-31-two-gpu-production-candidate.md
Task 4 Step 5) and produce predictions.json for evaluate-computer-vision-phase2.py.

Records per-request latency. GPU utilization/memory is sampled separately on
the host (this script has no nvidia-smi / GPU access inside its container)
via sample_gpu_host.sh, run in parallel and merged afterward.
"""
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import requests
import yaml

MANIFEST = Path("/repo/docs/computer-vision/phase-2-eval-manifest.yaml")
CORPUS_ROOT = Path("/corpus")
OUT_PREDICTIONS = Path("/corpus/_downloads/predictions.json")
OUT_METRICS = Path("/corpus/_downloads/runtime_metrics.json")

OCR_URL = "http://vision-ocr:8120/generate/ocr/paddle"
RTDETR_URL = "http://vision-detection:8121/generate/detect/rtdetr"
YOLOX_URL = "http://vision-yolox:8125/generate/detect/yolox"
GDINO_URL = "http://vision-grounding-dino:8124/generate/detect/grounding-dino"

LANG_FOR_GROUP = {"ocr_en": "en", "ocr_fi": "fi", "ocr_sv": "sv", "ocr_mixed": "auto"}

latencies_ms: dict[str, list[float]] = {"ocr": [], "rtdetr": [], "yolox": [], "grounding_dino": []}


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call(url: str, payload: dict, key: str) -> dict:
    started = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=120)
    elapsed_ms = (time.perf_counter() - started) * 1000
    latencies_ms[key].append(elapsed_ms)
    resp.raise_for_status()
    return resp.json()


def ocr_text_from_words(words: list[dict]) -> str:
    # Same top-to-bottom / left-to-right reconstruction used for the ocr_en
    # ground truth (reading_order_text in build_ocr_en.py), so predicted and
    # expected text are ordered comparably for CER/WER.
    ordered = sorted(words, key=lambda w: (round(w["box"]["y"] / 15), w["box"]["x"]))
    return " ".join(w["text"] for w in ordered)


def main():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    predictions: dict[str, dict] = {}

    total = len(entries)
    for i, entry in enumerate(entries, 1):
        groups = entry.get("groups", [])
        image_path = CORPUS_ROOT / entry["host_path"]
        image_b64 = b64_image(image_path)

        if any(g in LANG_FOR_GROUP for g in groups):
            group = next(g for g in groups if g in LANG_FOR_GROUP)
            result = call(OCR_URL, {"image": image_b64, "language": LANG_FOR_GROUP[group]}, "ocr")
            predictions[entry["id"]] = {"text": ocr_text_from_words(result["words"])}

        elif "detection_known_class" in groups:
            rtdetr = call(RTDETR_URL, {"image": image_b64, "confidence_threshold": 0.5}, "rtdetr")
            yolox = call(YOLOX_URL, {"image": image_b64, "confidence_threshold": 0.5}, "yolox")
            to_boxes = lambda dets: [
                {"category": d["label"], "bbox": [d["box"]["x"], d["box"]["y"], d["box"]["width"], d["box"]["height"]], "score": d["confidence"]}
                for d in dets["detections"]
            ]
            predictions[entry["id"]] = {
                "rtdetr": {"boxes": to_boxes(rtdetr)},
                "yolox": {"boxes": to_boxes(yolox)},
            }

        elif "detection_open_vocabulary" in groups:
            # LVIS category names are underscore_joined; Grounding-DINO's text
            # encoder expects natural-language phrases.
            labels = sorted({b["category"].replace("_", " ") for b in entry.get("expected_boxes", [])})
            gdino = call(GDINO_URL, {"image": image_b64, "labels": labels, "confidence_threshold": 0.4}, "grounding_dino")
            boxes = [
                {"category": d["label"], "bbox": [d["box"]["x"], d["box"]["y"], d["box"]["width"], d["box"]["height"]], "score": d["confidence"]}
                for d in gdino["detections"]
            ]
            predictions[entry["id"]] = {"grounding_dino": {"boxes": boxes}}

        else:
            raise ValueError(f"{entry['id']}: unrecognized group set {groups}")

        if i % 25 == 0 or i == total:
            print(f"  {i}/{total} entries predicted", file=sys.stderr)

    OUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREDICTIONS.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    def pct(values, p):
        if not values:
            return None
        s = sorted(values)
        k = int(round((p / 100) * (len(s) - 1)))
        return s[k]

    latency_summary = {
        key: {
            "count": len(values),
            "mean_ms": statistics.mean(values) if values else None,
            "p50_ms": pct(values, 50),
            "p95_ms": pct(values, 95),
            "max_ms": max(values) if values else None,
        }
        for key, values in latencies_ms.items()
    }
    metrics = {
        "latency": latency_summary,
        "total_entries": total,
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
