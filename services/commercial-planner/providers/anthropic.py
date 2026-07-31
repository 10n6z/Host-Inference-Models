"""Anthropic Messages API adapter for commercial vision planning.

Uses the official `anthropic` SDK with a forced strict tool call so the
response is schema-valid VisionPlan JSON, never free text to parse. Accepts
end_turn (direct structured output, if the model chose not to call the tool)
or tool_use (the forced tool); rejects truncation, refusal, or any other
stop reason as a typed PlannerError.
"""

from __future__ import annotations

import os

import anthropic

from metadata import PlanningMetadata
from vision_plan import VISION_PLAN_JSON_SCHEMA, PlannerError, VisionPlan, parse_vision_plan

# Deployment-configured allowlisted model -- never derived from client input.
DEFAULT_MODEL = os.environ.get("ANTHROPIC_PLANNER_MODEL", "claude-opus-5")
MAX_TOKENS = 1024

VISION_PLAN_TOOL = {
    "name": "emit_vision_plan",
    "description": "Emit the computer vision task plan for this request.",
    "input_schema": VISION_PLAN_JSON_SCHEMA,
    "strict": True,
}

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def plan_with_anthropic(metadata: PlanningMetadata) -> VisionPlan:
    client = _get_client()
    try:
        message = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            tools=[VISION_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "emit_vision_plan"},
            messages=[
                {
                    "role": "user",
                    "content": metadata.model_dump_json(),
                }
            ],
        )
    except anthropic.APIConnectionError as exc:
        raise PlannerError("PROVIDER_TIMEOUT", str(exc)) from exc
    except anthropic.APIStatusError as exc:
        raise PlannerError("PROVIDER_UNAVAILABLE", str(exc)) from exc

    if message.stop_reason == "refusal":
        raise PlannerError("REFUSAL", "Anthropic planner declined the request")
    if message.stop_reason == "max_tokens":
        raise PlannerError("TRUNCATED", "Anthropic planner output was truncated")

    if message.stop_reason == "tool_use":
        tool_block = next(
            (block for block in message.content if block.type == "tool_use"),
            None,
        )
        if tool_block is None:
            raise PlannerError("SCHEMA_INVALID", "No tool_use block in response")
        raw = tool_block.input
    elif message.stop_reason == "end_turn":
        text_block = next(
            (block for block in message.content if block.type == "text"),
            None,
        )
        if text_block is None:
            raise PlannerError("SCHEMA_INVALID", "No structured output in response")
        import json

        try:
            raw = json.loads(text_block.text)
        except (ValueError, TypeError) as exc:
            raise PlannerError("SCHEMA_INVALID", "Response was not valid JSON") from exc
    else:
        raise PlannerError(
            "SCHEMA_INVALID", f"Unexpected stop_reason '{message.stop_reason}'"
        )

    plan = parse_vision_plan(raw)
    return plan.model_copy(update={"source": "anthropic"})
