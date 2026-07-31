from types import SimpleNamespace

import anthropic
import pytest

import providers.anthropic as anthropic_provider
from metadata import JobFixture, build_planning_metadata
from vision_plan import PlannerError


@pytest.fixture
def metadata():
    job = JobFixture(
        prompt="Read the label",
        image_count=1,
        requested_domain="general",
        requested_tasks=["ocr"],
        ocr_mode="paddleocr",
        ocr_language="en",
        detector_mode="auto",
        requested_labels=[],
    )
    return build_planning_metadata(job)


class _FakeMessages:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.messages = _FakeMessages(response, error)


def _tool_use_message(plan: dict):
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[SimpleNamespace(type="tool_use", input=plan)],
    )


def _end_turn_message(plan_json: str):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=plan_json)],
    )


VALID_PLAN = {
    "domain": "general",
    "tasks": ["ocr", "report"],
    "reason": "user asked to read text",
    "warnings": [],
}


def test_forced_tool_use_produces_a_valid_plan(monkeypatch, metadata):
    monkeypatch.setattr(
        anthropic_provider, "_client", _FakeClient(_tool_use_message(VALID_PLAN))
    )

    plan = anthropic_provider.plan_with_anthropic(metadata)

    assert plan.source == "anthropic"
    assert plan.tasks == ["ocr", "report"]


def test_end_turn_with_structured_text_is_accepted(monkeypatch, metadata):
    import json

    monkeypatch.setattr(
        anthropic_provider,
        "_client",
        _FakeClient(_end_turn_message(json.dumps(VALID_PLAN))),
    )

    plan = anthropic_provider.plan_with_anthropic(metadata)

    assert plan.domain == "general"


def test_refusal_raises_typed_error(monkeypatch, metadata):
    monkeypatch.setattr(
        anthropic_provider,
        "_client",
        _FakeClient(SimpleNamespace(stop_reason="refusal", content=[])),
    )

    with pytest.raises(PlannerError) as exc_info:
        anthropic_provider.plan_with_anthropic(metadata)
    assert exc_info.value.code == "REFUSAL"


def test_truncation_raises_typed_error(monkeypatch, metadata):
    monkeypatch.setattr(
        anthropic_provider,
        "_client",
        _FakeClient(SimpleNamespace(stop_reason="max_tokens", content=[])),
    )

    with pytest.raises(PlannerError) as exc_info:
        anthropic_provider.plan_with_anthropic(metadata)
    assert exc_info.value.code == "TRUNCATED"


def test_invalid_plan_schema_raises_typed_error(monkeypatch, metadata):
    monkeypatch.setattr(
        anthropic_provider,
        "_client",
        _FakeClient(_tool_use_message({"domain": "not-a-real-domain", "tasks": []})),
    )

    with pytest.raises(PlannerError) as exc_info:
        anthropic_provider.plan_with_anthropic(metadata)
    assert exc_info.value.code == "SCHEMA_INVALID"


def test_connection_error_raises_provider_timeout(monkeypatch, metadata):
    error = anthropic.APIConnectionError(request=SimpleNamespace())
    monkeypatch.setattr(anthropic_provider, "_client", _FakeClient(error=error))

    with pytest.raises(PlannerError) as exc_info:
        anthropic_provider.plan_with_anthropic(metadata)
    assert exc_info.value.code == "PROVIDER_TIMEOUT"
