from __future__ import annotations

import os
from typing import Any

import httpx


class VllmAdapterError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _resolve_messages(
    input_payload: dict[str, Any], parameters: dict[str, Any]
) -> list[dict[str, str]]:
    messages = input_payload.get("messages")
    if isinstance(messages, list) and messages:
        resolved = [
            {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
            for m in messages
            if isinstance(m, dict) and str(m.get("content") or "").strip()
        ]
        if resolved:
            return resolved

    prompt = input_payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VllmAdapterError("Prompt is required for vLLM text generation.", 422)

    system_instruction = parameters.get("system_instruction")
    result = []
    if isinstance(system_instruction, str) and system_instruction.strip():
        result.append({"role": "system", "content": system_instruction.strip()})
    result.append({"role": "user", "content": prompt.strip()})
    return result


def _sampling_params(parameters: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if isinstance(parameters.get("temperature"), (int, float)):
        params["temperature"] = parameters["temperature"]
    max_tokens = parameters.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        params["max_tokens"] = max_tokens
    if isinstance(parameters.get("top_p"), (int, float)):
        params["top_p"] = parameters["top_p"]
    return params


async def generate(
    *,
    model_id: str,
    entry: dict[str, Any],
    raw_body: dict[str, Any],
    timeout_seconds: float,
    request_id: str,
) -> dict[str, Any]:
    input_payload = raw_body.get("input") if isinstance(raw_body.get("input"), dict) else {}
    parameters = raw_body.get("parameters") if isinstance(raw_body.get("parameters"), dict) else {}
    messages = _resolve_messages(input_payload, parameters)

    runtime_model = entry.get("runtime_model") or model_id
    request_body = {
        "model": runtime_model,
        "messages": messages,
        **_sampling_params(parameters),
    }

    # VLLM_BASE_URL overrides the registry endpoint for deployments where the
    # vLLM OpenAI-compatible server isn't the compose-internal service.
    endpoint = str(entry["endpoint"])
    base_override = os.getenv("VLLM_BASE_URL", "").strip().rstrip("/")
    if base_override:
        endpoint = f"{base_override}/v1/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=False) as client:
            response = await client.post(endpoint, json=request_body, headers={"X-Request-ID": request_id})
    except httpx.TimeoutException as exc:
        raise VllmAdapterError(f"Request timed out after {int(timeout_seconds)} seconds.", 504) from exc
    except httpx.RequestError as exc:
        raise VllmAdapterError(f"Could not reach vLLM runtime: {exc}", 503) from exc

    if response.status_code >= 400:
        raise VllmAdapterError(
            f"vLLM returned HTTP {response.status_code}: {response.text[:500]}",
            response.status_code,
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise VllmAdapterError("vLLM returned invalid JSON.") from exc

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VllmAdapterError("vLLM response did not include any choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    output_text = message.get("content") if isinstance(message, dict) else None
    finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
    if not isinstance(output_text, str) or not output_text.strip():
        if finish_reason == "length":
            raise VllmAdapterError(
                "Token limit reached before the model produced a final answer. "
                "Increase max_tokens.",
                422,
            )
        raise VllmAdapterError("vLLM response did not include generated text.")

    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return {
        "success": True,
        "status": "completed",
        "model": model_id,
        "model_id": model_id,
        "output_text": output_text,
        "parameters_used": {k: v for k, v in request_body.items() if k not in {"model", "messages"}},
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "metadata": {
            "provider": "vllm",
            "runtime_model": runtime_model,
            "finish_reason": finish_reason,
        },
    }
