from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


ISSUER = "sw4e-control-plane"
AUDIENCE = "model-gateway"
bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class GatewayClaims:
    user_id: str
    tenant_id: str
    permitted_tasks: tuple[str, ...]
    permitted_model_ids: tuple[str, ...]
    job_id: str | None = None


class GatewayAuthError(Exception):
    def __init__(self, status_code: int, code: str):
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def _decode_json(segment: str) -> dict[str, object]:
    padding = "=" * (-len(segment) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{segment}{padding}")
        value = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID") from error
    if not isinstance(value, dict):
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID")
    return value


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID")
    return tuple(value)


def verify_gateway_jwt(token: str, secret: str) -> GatewayClaims:
    parts = token.split(".")
    if len(parts) != 3 or not secret:
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID")
    header = _decode_json(parts[0])
    if header.get("alg") != "HS256":
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID")
    expected = hmac.new(
        secret.encode(), f"{parts[0]}.{parts[1]}".encode(), hashlib.sha256
    ).digest()
    supplied_padding = "=" * (-len(parts[2]) % 4)
    try:
        supplied = base64.urlsafe_b64decode(f"{parts[2]}{supplied_padding}")
    except ValueError as error:
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID") from error
    if not hmac.compare_digest(expected, supplied):
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID")
    payload = _decode_json(parts[1])
    if (
        payload.get("iss") != ISSUER
        or payload.get("aud") != AUDIENCE
        or not isinstance(payload.get("exp"), int)
        or payload["exp"] <= time.time()
        or not isinstance(payload.get("userId"), str)
        or not payload["userId"]
        or not isinstance(payload.get("tenantId"), str)
        or not payload["tenantId"]
    ):
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID")
    job_id = payload.get("jobId")
    if job_id is not None and (not isinstance(job_id, str) or not job_id):
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID")
    return GatewayClaims(
        user_id=payload["userId"],
        tenant_id=payload["tenantId"],
        permitted_tasks=_string_list(payload.get("permittedTasks")),
        permitted_model_ids=_string_list(payload.get("permittedModelIds")),
        job_id=job_id,
    )


def require_gateway_scope(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> GatewayClaims:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise GatewayAuthError(401, "GATEWAY_TOKEN_INVALID")
    return verify_gateway_jwt(
        credentials.credentials,
        os.getenv("MODEL_GATEWAY_JWT_SECRET", ""),
    )


def require_model_scope(claims: GatewayClaims, task: str, model_id: str) -> None:
    if task not in claims.permitted_tasks or model_id not in claims.permitted_model_ids:
        raise GatewayAuthError(403, "GATEWAY_SCOPE_DENIED")
