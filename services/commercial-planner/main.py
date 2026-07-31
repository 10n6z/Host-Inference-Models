"""commercial-planner -- metadata-only OpenAI/Anthropic vision planning.

Never returns provider keys or key-presence metadata. Deployment defaults
disable both providers (PLANNER_PROVIDERS_ENABLED=self_hosted,rules); the
control-plane's own tenant policy is the second, independent gate.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from metadata import JobFixture, build_planning_metadata
from providers.anthropic import plan_with_anthropic
from providers.openai import plan_with_openai
from vision_plan import PlannerError

logger = logging.getLogger("commercial-planner")
logging.basicConfig(level=os.environ.get("COMMERCIAL_PLANNER_LOG_LEVEL", "info").upper())

app = FastAPI(title="commercial-planner", version="1.0.0")

PLANNER_STATUS_CODES = {
    "REFUSAL": 422,
    "TRUNCATED": 502,
    "SCHEMA_INVALID": 502,
    "PROVIDER_TIMEOUT": 504,
    "PROVIDER_UNAVAILABLE": 503,
}


class PlanRequest(BaseModel):
    prompt: str
    image_count: int
    requested_domain: str
    requested_tasks: list[str]
    ocr_mode: str
    ocr_language: str
    detector_mode: str
    requested_labels: list[str] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "commercial-planner"}


def _run_plan(request: PlanRequest, planner):
    job = JobFixture(
        prompt=request.prompt,
        image_count=request.image_count,
        requested_domain=request.requested_domain,
        requested_tasks=request.requested_tasks,
        ocr_mode=request.ocr_mode,
        ocr_language=request.ocr_language,
        detector_mode=request.detector_mode,
        requested_labels=request.requested_labels,
    )
    metadata = build_planning_metadata(job)
    try:
        plan = planner(metadata)
    except PlannerError as exc:
        status = PLANNER_STATUS_CODES.get(exc.code, 502)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)}) from exc
    return plan.model_dump()


@app.post("/generate/plan/openai")
def generate_plan_openai(request: PlanRequest) -> dict:
    return _run_plan(request, plan_with_openai)


@app.post("/generate/plan/anthropic")
def generate_plan_anthropic(request: PlanRequest) -> dict:
    return _run_plan(request, plan_with_anthropic)


@app.exception_handler(HTTPException)
def _http_exception_handler(_, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "ERROR", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=detail)
