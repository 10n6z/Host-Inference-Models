"""OpenAI Responses API adapter for commercial vision planning.

Uses the official `openai` SDK with strict JSON Schema output so the
response is schema-valid VisionPlan JSON. Treats refusal, incomplete
output, and schema mismatch as typed failures.
"""

from __future__ import annotations

import json
import os

import openai

from metadata import PlanningMetadata
from vision_plan import VISION_PLAN_JSON_SCHEMA, PlannerError, VisionPlan, parse_vision_plan

# Deployment-configured allowlisted model -- never derived from client input.
DEFAULT_MODEL = os.environ.get("OPENAI_PLANNER_MODEL", "gpt-5.6-luna")

VISION_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "vision_plan",
    "schema": VISION_PLAN_JSON_SCHEMA,
    "strict": True,
}

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


def plan_with_openai(metadata: PlanningMetadata) -> VisionPlan:
    client = _get_client()
    try:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            reasoning={"effort": "low"},
            input=metadata.model_dump_json(),
            text={"format": VISION_PLAN_RESPONSE_FORMAT},
        )
    except openai.APITimeoutError as exc:
        raise PlannerError("PROVIDER_TIMEOUT", str(exc)) from exc
    except openai.APIStatusError as exc:
        raise PlannerError("PROVIDER_UNAVAILABLE", str(exc)) from exc

    if getattr(response, "status", None) == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
        if reason == "content_filter":
            raise PlannerError("REFUSAL", "OpenAI planner declined the request")
        raise PlannerError("TRUNCATED", f"OpenAI planner output was incomplete ({reason})")

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise PlannerError("SCHEMA_INVALID", "No structured output in response")

    try:
        raw = json.loads(output_text)
    except (ValueError, TypeError) as exc:
        raise PlannerError("SCHEMA_INVALID", "Response was not valid JSON") from exc

    plan = parse_vision_plan(raw)
    return plan.model_copy(update={"source": "openai"})
