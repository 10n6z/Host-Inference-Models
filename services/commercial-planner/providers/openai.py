"""OpenAI Responses API adapter for commercial vision planning.

Uses the official `openai` SDK with strict JSON Schema output so the
response is schema-valid VisionPlan JSON. Treats refusal, incomplete
output, and schema mismatch as typed failures.

If OPENROUTER_API_KEY is set, routes through OpenRouter's Chat Completions
endpoint instead -- OpenRouter has no Responses API, so this is a genuinely
different call shape, not just a different base_url. Used for staging/test
tenants that need a free-tier external-provider path (Task 18 Step 6);
`source` on the returned plan stays "openai" since the tenant still selects
the "openai" PlannerProvider -- OpenRouter is a deployment substitution
behind that choice, the same pattern as the Ministral -> Mistral-7B
substitution in Phase 2.
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

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# OpenRouter's free-tier model roster changes over time (verified 2026-08-01:
# meta-llama/llama-3.1-8b-instruct:free had been retired to paid-only since
# this default was first picked). Override with OPENROUTER_PLANNER_MODEL if
# this one is retired too -- check https://openrouter.ai/api/v1/models for
# current `:free`-suffixed entries.
OPENROUTER_MODEL = os.environ.get("OPENROUTER_PLANNER_MODEL", "openai/gpt-oss-20b:free")
OPENROUTER_SYSTEM_PROMPT = (
    "You are a Computer Vision job planner. Given compact task metadata, "
    "respond with ONLY a single JSON object matching this exact schema -- "
    "no prose, no markdown code fences, no extra keys: "
    + json.dumps(VISION_PLAN_JSON_SCHEMA)
)

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        if OPENROUTER_API_KEY:
            _client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        else:
            _client = openai.OpenAI()
    return _client


def plan_with_openai(metadata: PlanningMetadata) -> VisionPlan:
    client = _get_client()
    if OPENROUTER_API_KEY:
        return _plan_via_chat_completions(client, metadata)
    return _plan_via_responses_api(client, metadata)


def _plan_via_responses_api(client: openai.OpenAI, metadata: PlanningMetadata) -> VisionPlan:
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


def _plan_via_chat_completions(client: openai.OpenAI, metadata: PlanningMetadata) -> VisionPlan:
    # OpenRouter (and Chat Completions generally) has no native "strict JSON
    # Schema" mode as uniformly as the Responses API -- response_format is
    # loosened to json_object and parse_vision_plan() below does the actual
    # schema enforcement, same typed-failure contract either way.
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": OPENROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": metadata.model_dump_json()},
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )
    except openai.APITimeoutError as exc:
        raise PlannerError("PROVIDER_TIMEOUT", str(exc)) from exc
    except openai.APIStatusError as exc:
        raise PlannerError("PROVIDER_UNAVAILABLE", str(exc)) from exc

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise PlannerError("SCHEMA_INVALID", "No choices in response")
    choice = choices[0]

    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "content_filter":
        raise PlannerError("REFUSAL", "OpenRouter planner declined the request")
    if finish_reason == "length":
        raise PlannerError("TRUNCATED", "OpenRouter planner output was truncated")

    output_text = getattr(getattr(choice, "message", None), "content", None)
    if not output_text:
        raise PlannerError("SCHEMA_INVALID", "No structured output in response")

    try:
        raw = json.loads(output_text)
    except (ValueError, TypeError) as exc:
        raise PlannerError("SCHEMA_INVALID", "Response was not valid JSON") from exc

    plan = parse_vision_plan(raw)
    return plan.model_copy(update={"source": "openai"})
