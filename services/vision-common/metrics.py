"""Shared Prometheus instrumentation for vision-* FastAPI services.

Each service imports build_vision_metrics(service_name) once at import time
and calls .observe_inference(model_id) as a context manager around its
inference call, then mounts .render() at GET /metrics.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


class VisionServiceMetrics:
    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self.registry = CollectorRegistry()
        self.inference_duration = Histogram(
            "sw4e_vision_adapter_duration_seconds",
            "Vision adapter inference duration in seconds",
            labelnames=("service", "model_id", "status"),
            registry=self.registry,
        )
        self.inference_total = Counter(
            "sw4e_vision_adapter_requests_total",
            "Vision adapter inference requests",
            labelnames=("service", "model_id", "status"),
            registry=self.registry,
        )

    @contextmanager
    def observe_inference(self, model_id: str) -> Iterator[dict]:
        """Times a block and records status="completed" unless the block
        raises, in which case status="failed" is recorded and the
        exception is re-raised unchanged."""
        ctx = {"status": "completed"}
        start = time.perf_counter()
        try:
            yield ctx
        except Exception:
            ctx["status"] = "failed"
            raise
        finally:
            duration = time.perf_counter() - start
            labels = (self.service_name, model_id, ctx["status"])
            self.inference_duration.labels(*labels).observe(duration)
            self.inference_total.labels(*labels).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


def build_vision_metrics(service_name: str) -> VisionServiceMetrics:
    return VisionServiceMetrics(service_name)


VISION_METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST
