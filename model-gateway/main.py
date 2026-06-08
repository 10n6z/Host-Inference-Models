from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICES_ROOT = REPO_ROOT / "services"
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

from common.responses import build_failed_response, build_success_response, utc_now_iso

load_dotenv(REPO_ROOT / "servers" / ".env")
load_dotenv(REPO_ROOT / ".env")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", Path(__file__).parent / "registry.yaml"))
GATEWAY_TIMEOUT_SECONDS = float(os.getenv("GATEWAY_TIMEOUT_SECONDS", "300"))
GATEWAY_PUBLIC_BASE_URL = os.getenv("GATEWAY_PUBLIC_BASE_URL", "http://localhost:9000").rstrip("/")

app = FastAPI(title="Model Server Gateway")

def _load_registry() -> dict[str, Any]:
    with open(REGISTRY_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _registry_models() -> dict[str, dict[str, Any]]:
    registry = _load_registry()
    return registry.get("models") or {}


def _registry_services() -> dict[str, dict[str, Any]]:
    registry = _load_registry()
    return registry.get("services") or {}


def _models_payload() -> list[dict[str, Any]]:
    models = []
    for model_id, entry in sorted(_registry_models().items()):
        models.append(
            {
                "id": model_id,
                "displayName": entry.get("display_name", model_id),
                "modality": entry.get("modality", "text-to-speech"),
                "service": entry.get("service"),
                "family": entry.get("family"),
                "endpoint": entry.get("endpoint"),
                "fields": entry.get("fields") or {},
            }
        )
    return models


def _map_http_status_to_error_type(status_code: int, upstream_body: dict[str, Any]) -> str:
    if status_code == 422:
        return "ValidationError"
    if status_code in (503, 504):
        return "ServiceUnavailable"
    upstream_code = str(upstream_body.get("code", "")).upper()
    if "MODEL" in upstream_code and "LOAD" in upstream_code:
        return "ModelLoadError"
    if status_code >= 500:
        return "InferenceError"
    return "ValidationError"


def _translate_upstream_error(
    *,
    model: str,
    entry: dict[str, Any],
    request_id: str,
    status_code: int,
    upstream_body: dict[str, Any],
) -> JSONResponse:
    message = upstream_body.get("message") or upstream_body.get("error", {}).get("message") or "Upstream request failed."
    error_type = _map_http_status_to_error_type(status_code, upstream_body)
    payload = build_failed_response(
        model=model,
        service=entry.get("service", "unknown"),
        family=entry.get("family", "unknown"),
        error_type=error_type,
        message=str(message),
        request_id=request_id,
        details=upstream_body.get("details") or upstream_body.get("error", {}).get("details") or {},
    )
    return JSONResponse(status_code=status_code, content=payload)


def _request_id_from(raw_body: dict[str, Any], request: Request) -> str:
    header_request_id = request.headers.get("X-Request-ID")
    if header_request_id:
        return header_request_id
    body_request_id = raw_body.get("request_id")
    if body_request_id is not None:
        body_request_id = str(body_request_id).strip()
        if body_request_id:
            return body_request_id
    return uuid.uuid4().hex


def _forward_payload_from(raw_body: dict[str, Any], *, model_id: str, include_model: bool) -> dict[str, Any]:
    nested_input = raw_body.get("input")
    nested_parameters = raw_body.get("parameters")
    uses_nested_contract = "input" in raw_body or "parameters" in raw_body

    if not uses_nested_contract:
        forward_payload = {key: value for key, value in raw_body.items() if key not in {"model", "request_id"}}
    else:
        if nested_input is None:
            nested_input = {}
        if nested_parameters is None:
            nested_parameters = {}
        if not isinstance(nested_input, dict):
            raise ValueError("input: Must be an object")
        if not isinstance(nested_parameters, dict):
            raise ValueError("parameters: Must be an object")

        forward_payload = {
            key: value
            for key, value in raw_body.items()
            if key not in {"model", "input", "parameters", "request_id"}
        }
        forward_payload.update(nested_input)
        forward_payload.update(nested_parameters)

    if include_model:
        forward_payload["model"] = model_id
    return forward_payload


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(part) for part in first.get("loc", []) if part not in ("body",))
    message = first.get("msg", "Invalid request body.")
    if loc:
        message = f"{loc}: {message}"
    payload = build_failed_response(
        model="unknown",
        service="model-gateway",
        family="model-gateway",
        error_type="ValidationError",
        message=message,
        request_id=request_id,
        details={"errors": exc.errors()},
    )
    return JSONResponse(status_code=422, content=payload)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "model-gateway",
        "registry_path": str(REGISTRY_PATH),
        "created_at": utc_now_iso(),
    }


@app.get("/models")
def models():
    return {
        "models": _models_payload(),
        "services": _registry_services(),
        "gateway_public_base_url": GATEWAY_PUBLIC_BASE_URL,
    }


@app.post("/generate")
async def generate(request: Request):
    raw_body = await request.json()
    request_id = _request_id_from(raw_body if isinstance(raw_body, dict) else {}, request)
    if not isinstance(raw_body, dict):
        payload = build_failed_response(
            model="unknown",
            service="model-gateway",
            family="model-gateway",
            error_type="ValidationError",
            message="Request body must be a JSON object.",
            request_id=request_id,
        )
        return JSONResponse(status_code=422, content=payload)

    model_id = str(raw_body.get("model", "")).strip()
    if not model_id:
        payload = build_failed_response(
            model="unknown",
            service="model-gateway",
            family="model-gateway",
            error_type="ValidationError",
            message="model: Field required",
            request_id=request_id,
        )
        return JSONResponse(status_code=422, content=payload)

    registry_entry = _registry_models().get(model_id)

    if registry_entry is None:
        payload = build_failed_response(
            model=model_id,
            service="model-gateway",
            family="model-gateway",
            error_type="UnknownModel",
            message=f"Unknown model '{model_id}'.",
            request_id=request_id,
            details={"available_models": sorted(_registry_models().keys())},
        )
        return JSONResponse(status_code=404, content=payload)

    target_endpoint = registry_entry["endpoint"]
    family = registry_entry.get("family", registry_entry.get("service", "unknown"))
    service_name = registry_entry.get("service", "unknown")

    try:
        forward_payload = _forward_payload_from(
            raw_body,
            model_id=model_id,
            include_model=target_endpoint.endswith("/generate"),
        )
    except ValueError as exc:
        payload = build_failed_response(
            model=model_id,
            service="model-gateway",
            family="model-gateway",
            error_type="ValidationError",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(status_code=422, content=payload)

    started = time.perf_counter()
    logger.info(
        "gateway_request request_id=%s model=%s service=%s endpoint=%s",
        request_id,
        model_id,
        service_name,
        target_endpoint,
    )

    try:
        async with httpx.AsyncClient(timeout=GATEWAY_TIMEOUT_SECONDS, trust_env=False) as client:
            upstream = await client.post(
                target_endpoint,
                json=forward_payload,
                headers={"X-Request-ID": request_id},
            )
    except httpx.TimeoutException:
        payload = build_failed_response(
            model=model_id,
            service=service_name,
            family=family,
            error_type="ServiceUnavailable",
            message=f"Request timed out after {int(GATEWAY_TIMEOUT_SECONDS)} seconds.",
            request_id=request_id,
        )
        return JSONResponse(status_code=504, content=payload)
    except httpx.RequestError as exc:
        payload = build_failed_response(
            model=model_id,
            service=service_name,
            family=family,
            error_type="ServiceUnavailable",
            message=f"Could not reach service '{service_name}': {exc}",
            request_id=request_id,
        )
        return JSONResponse(status_code=503, content=payload)

    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "gateway_response request_id=%s model=%s service=%s status=%s duration_ms=%s",
        request_id,
        model_id,
        service_name,
        upstream.status_code,
        duration_ms,
    )

    try:
        upstream_body = upstream.json()
    except ValueError:
        upstream_body = {"message": upstream.text}

    if upstream.status_code >= 400:
        return _translate_upstream_error(
            model=model_id,
            entry=registry_entry,
            request_id=request_id,
            status_code=upstream.status_code,
            upstream_body=upstream_body if isinstance(upstream_body, dict) else {"message": str(upstream_body)},
        )

    if not isinstance(upstream_body, dict):
        upstream_body = {"result": upstream_body}

    success_payload = build_success_response(upstream_body, request_id=request_id, family=family)
    success_payload.setdefault("model_id", model_id)
    success_payload.setdefault("modelId", model_id)
    success_payload["duration_ms"] = duration_ms
    return JSONResponse(status_code=200, content=success_payload)
