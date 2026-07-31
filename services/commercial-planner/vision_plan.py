"""Shared VisionPlan contract for every commercial planner adapter.

Matches the shape validated by
sandbox/control-plane/src/services/computer-vision/schemas.ts's
visionPlanSchema, so a commercial-provider plan slots into the same pipeline
as the rules/self-hosted planners without a translation layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ALLOWED_TASKS = {
    "ocr",
    "detection",
    "comparison",
    "ui_analysis",
    "safety_analysis",
    "counting",
    "annotate",
    "report",
}
ALLOWED_DOMAINS = {"software_visual_qa", "operations_safety", "general"}


class VisionPlan(BaseModel):
    domain: str
    tasks: list[str]
    reason: str
    source: str
    warnings: list[str] = Field(default_factory=list)


class PlannerError(Exception):
    """Typed failure for a commercial planner call.

    code is one of REFUSAL, TRUNCATED, SCHEMA_INVALID, PROVIDER_TIMEOUT,
    PROVIDER_UNAVAILABLE -- callers branch on this, never on message text.
    """

    def __init__(self, code: str, message: str | None = None):
        super().__init__(message or code)
        self.code = code


def parse_vision_plan(raw: dict) -> VisionPlan:
    domain = raw.get("domain")
    tasks = raw.get("tasks")
    if domain not in ALLOWED_DOMAINS:
        raise PlannerError("SCHEMA_INVALID", f"Unknown domain '{domain}'")
    if not isinstance(tasks, list) or not tasks or not all(
        isinstance(task, str) and task in ALLOWED_TASKS for task in tasks
    ):
        raise PlannerError("SCHEMA_INVALID", "Plan tasks failed validation")
    return VisionPlan(
        domain=domain,
        tasks=tasks,
        reason=str(raw.get("reason", ""))[:500],
        source=str(raw.get("source", "")),
        warnings=[str(w) for w in raw.get("warnings", [])],
    )


VISION_PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string", "enum": sorted(ALLOWED_DOMAINS)},
        "tasks": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(ALLOWED_TASKS)},
            "minItems": 1,
        },
        "reason": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["domain", "tasks", "reason", "warnings"],
    "additionalProperties": False,
}
