import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_common_metrics import build_vision_metrics


def test_records_completed_inference():
    m = build_vision_metrics("vision-detection")
    with m.observe_inference("rtdetr-r50vd"):
        pass
    body = m.render().decode()
    assert "sw4e_vision_adapter_requests_total" in body
    assert (
        'sw4e_vision_adapter_requests_total{model_id="rtdetr-r50vd",service="vision-detection",status="completed"} 1.0'
        in body
    )


def test_records_failed_inference_and_reraises():
    m = build_vision_metrics("vision-detection")
    raised = False
    try:
        with m.observe_inference("rtdetr-r50vd"):
            raise ValueError("boom")
    except ValueError:
        raised = True
    assert raised
    body = m.render().decode()
    assert (
        'sw4e_vision_adapter_requests_total{model_id="rtdetr-r50vd",service="vision-detection",status="failed"} 1.0'
        in body
    )


def test_separate_instances_do_not_share_registries():
    a = build_vision_metrics("vision-detection")
    b = build_vision_metrics("vision-yolox")
    with a.observe_inference("rtdetr-r50vd"):
        pass
    assert "vision-yolox" not in a.render().decode()
    assert "vision-detection" not in b.render().decode()
