"""Prometheus instrumentation for model-gateway.

There is no true internal queue in this gateway -- /generate proxies
synchronously to an upstream service or job-tracking adapter. The in-flight
gauge below is the honest proxy for "queue depth": concurrent /generate
calls the gateway is currently holding open.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, generate_latest

registry = CollectorRegistry()

requests_in_flight = Gauge(
    "sw4e_gateway_requests_in_flight",
    "Concurrent /generate requests the gateway is currently handling",
    registry=registry,
)

requests_total = Counter(
    "sw4e_gateway_requests_total",
    "Gateway /generate requests by model and outcome",
    labelnames=("model_id", "status"),
    registry=registry,
)

provider_errors_total = Counter(
    "sw4e_gateway_provider_errors_total",
    "Gateway /generate error responses by model and error type",
    labelnames=("model_id", "error_type"),
    registry=registry,
)


def record_response(model_id: str, status_code: int, error_type: str | None) -> None:
    outcome = "success" if status_code < 400 else "error"
    requests_total.labels(model_id=model_id, status=outcome).inc()
    if outcome == "error":
        provider_errors_total.labels(model_id=model_id, error_type=error_type or "Unknown").inc()


def render() -> bytes:
    return generate_latest(registry)


METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST
