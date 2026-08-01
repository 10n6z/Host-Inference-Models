import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics


def test_success_records_success_status():
    metrics.record_response("pp-ocrv4-test-success", 200, None)
    body = metrics.render().decode()
    assert (
        'sw4e_gateway_requests_total{model_id="pp-ocrv4-test-success",status="success"} 1.0'
        in body
    )


def test_error_records_error_status_and_type():
    metrics.record_response("rtdetr-test-error", 503, "ServiceUnavailable")
    body = metrics.render().decode()
    assert (
        'sw4e_gateway_requests_total{model_id="rtdetr-test-error",status="error"} 1.0'
        in body
    )
    assert (
        'sw4e_gateway_provider_errors_total{error_type="ServiceUnavailable",model_id="rtdetr-test-error"} 1.0'
        in body
    )


def test_success_does_not_increment_provider_errors():
    metrics.record_response("pp-ocrv4-test-clean", 200, None)
    body = metrics.render().decode()
    assert (
        'sw4e_gateway_provider_errors_total{error_type="Unknown",model_id="pp-ocrv4-test-clean"}'
        not in body
    )


def test_in_flight_gauge_tracks_concurrency():
    metrics.requests_in_flight.inc()
    metrics.requests_in_flight.inc()
    body = metrics.render().decode()
    assert "sw4e_gateway_requests_in_flight 2.0" in body
    metrics.requests_in_flight.dec()
    metrics.requests_in_flight.dec()
    body = metrics.render().decode()
    assert "sw4e_gateway_requests_in_flight 0.0" in body
