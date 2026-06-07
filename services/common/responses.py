from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_failed_response(
    *,
    model: str,
    service: str,
    family: str,
    error_type: str,
    message: str,
    request_id: str,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "model": model,
        "service": service,
        "error": {
            "message": message,
            "type": error_type,
            "details": details or {},
        },
        "metadata": {
            "request_id": request_id,
            "family": family,
        },
    }


def build_success_response(payload: dict[str, Any], *, request_id: str, family: str) -> dict[str, Any]:
    response = dict(payload)
    response.setdefault("status", "completed")
    metadata = dict(response.get("metadata") or {})
    metadata.setdefault("request_id", request_id)
    metadata.setdefault("family", family)
    response["metadata"] = metadata
    return response
