import json
from types import SimpleNamespace

import openai
import pytest

import providers.openai as openai_provider
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


class _FakeResponses:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class _FakeCompletions:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class _FakeChat:
    def __init__(self, response=None, error=None):
        self.completions = _FakeCompletions(response, error)


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.responses = _FakeResponses(response, error)
        self.chat = _FakeChat(response, error)


VALID_PLAN = {
    "domain": "general",
    "tasks": ["ocr", "report"],
    "reason": "user asked to read text",
    "warnings": [],
}


def test_successful_plan_is_parsed(monkeypatch, metadata):
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(SimpleNamespace(status="completed", output_text=json.dumps(VALID_PLAN))),
    )

    plan = openai_provider.plan_with_openai(metadata)

    assert plan.source == "openai"
    assert plan.tasks == ["ocr", "report"]


def test_content_filter_incomplete_status_is_a_refusal(monkeypatch, metadata):
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(
            SimpleNamespace(
                status="incomplete",
                output_text=None,
                incomplete_details=SimpleNamespace(reason="content_filter"),
            )
        ),
    )

    with pytest.raises(PlannerError) as exc_info:
        openai_provider.plan_with_openai(metadata)
    assert exc_info.value.code == "REFUSAL"


def test_max_output_tokens_incomplete_status_is_truncated(monkeypatch, metadata):
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(
            SimpleNamespace(
                status="incomplete",
                output_text=None,
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )
        ),
    )

    with pytest.raises(PlannerError) as exc_info:
        openai_provider.plan_with_openai(metadata)
    assert exc_info.value.code == "TRUNCATED"


def test_invalid_json_output_is_schema_invalid(monkeypatch, metadata):
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(SimpleNamespace(status="completed", output_text="not json")),
    )

    with pytest.raises(PlannerError) as exc_info:
        openai_provider.plan_with_openai(metadata)
    assert exc_info.value.code == "SCHEMA_INVALID"


def test_api_timeout_raises_provider_timeout(monkeypatch, metadata):
    error = openai.APITimeoutError(request=SimpleNamespace())
    monkeypatch.setattr(openai_provider, "_client", _FakeClient(error=error))

    with pytest.raises(PlannerError) as exc_info:
        openai_provider.plan_with_openai(metadata)
    assert exc_info.value.code == "PROVIDER_TIMEOUT"


def _chat_response(content=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_openrouter_path_parses_a_successful_plan(monkeypatch, metadata):
    monkeypatch.setattr(openai_provider, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(_chat_response(content=json.dumps(VALID_PLAN))),
    )

    plan = openai_provider.plan_with_openai(metadata)

    assert plan.source == "openai"
    assert plan.tasks == ["ocr", "report"]


def test_openrouter_path_treats_content_filter_as_refusal(monkeypatch, metadata):
    monkeypatch.setattr(openai_provider, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(_chat_response(content=None, finish_reason="content_filter")),
    )

    with pytest.raises(PlannerError) as exc_info:
        openai_provider.plan_with_openai(metadata)
    assert exc_info.value.code == "REFUSAL"


def test_openrouter_path_treats_length_finish_as_truncated(monkeypatch, metadata):
    monkeypatch.setattr(openai_provider, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(_chat_response(content=None, finish_reason="length")),
    )

    with pytest.raises(PlannerError) as exc_info:
        openai_provider.plan_with_openai(metadata)
    assert exc_info.value.code == "TRUNCATED"


def test_openrouter_path_rejects_non_json_output(monkeypatch, metadata):
    monkeypatch.setattr(openai_provider, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(_chat_response(content="not json")),
    )

    with pytest.raises(PlannerError) as exc_info:
        openai_provider.plan_with_openai(metadata)
    assert exc_info.value.code == "SCHEMA_INVALID"


def test_openrouter_path_rejects_schema_violating_json(monkeypatch, metadata):
    monkeypatch.setattr(openai_provider, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        openai_provider,
        "_client",
        _FakeClient(_chat_response(content=json.dumps({"domain": "not_a_real_domain"}))),
    )

    with pytest.raises(PlannerError) as exc_info:
        openai_provider.plan_with_openai(metadata)
    assert exc_info.value.code == "SCHEMA_INVALID"
