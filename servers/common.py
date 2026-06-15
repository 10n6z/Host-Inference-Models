from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: Optional[dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _field_spec(
    field_type: str,
    *,
    required: Optional[bool] = None,
    default: Any = None,
    minimum: Any = None,
    maximum: Any = None,
    step: Any = None,
    enum: Optional[list[Any]] = None,
    max_length: Optional[int] = None,
    description: Optional[str] = None,
):
    data: dict[str, Any] = {"type": field_type}
    if required is not None:
        data["required"] = required
    if default is not None:
        data["default"] = default
    if minimum is not None:
        data["min"] = minimum
    if maximum is not None:
        data["max"] = maximum
    if step is not None:
        data["step"] = step
    if enum is not None:
        data["enum"] = enum
    if max_length is not None:
        data["max_length"] = max_length
    if description:
        data["description"] = description
    return data


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _check_output_exists(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise APIError(
            code="OUTPUT_SAVE_FAILED",
            message="Model run completed but output file was not saved.",
            status_code=500,
            details={"output_path": str(path)},
        )


def _map_runtime_error(exc: Exception) -> APIError:
    message = str(exc)
    lower = message.lower()
    if isinstance(exc, APIError):
        return exc
    if isinstance(exc, FileNotFoundError) or "model folder not found" in lower:
        return APIError("MODEL_NOT_LOADED", message, 503)
    if "cuda out of memory" in lower or "out of memory" in lower:
        return APIError("CUDA_OUT_OF_MEMORY", message, 507)
    return APIError("GENERATION_FAILED", message, 500)


def _validation_message(errors: list[dict[str, Any]]) -> str:
    first = errors[0] if errors else {}
    loc = first.get("loc", [])
    loc_text = ".".join(str(part) for part in loc if part not in ("body",))
    msg = first.get("msg", "Invalid request body.")
    if loc_text:
        return f"{loc_text}: {msg}"
    return msg
