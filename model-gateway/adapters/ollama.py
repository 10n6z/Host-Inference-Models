from __future__ import annotations

import os
from typing import Any

import httpx


class OllamaAdapterError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().title()
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _resolve_prompt(input_payload: dict[str, Any], parameters: dict[str, Any]) -> str:
    prompt = input_payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        resolved = prompt.strip()
    else:
        messages = input_payload.get("messages")
        resolved = _prompt_from_messages([m for m in messages if isinstance(m, dict)]) if isinstance(messages, list) else ""

    system_instruction = parameters.get("system_instruction")
    if isinstance(system_instruction, str) and system_instruction.strip():
        return f"System: {system_instruction.strip()}\n\n{resolved}".strip()
    return resolved


def _ollama_options(parameters: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if isinstance(parameters.get("temperature"), (int, float)):
        options["temperature"] = parameters["temperature"]
    max_tokens = parameters.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        options["num_predict"] = max_tokens
    if isinstance(parameters.get("top_p"), (int, float)):
        options["top_p"] = parameters["top_p"]
    return options


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
    prompt = _resolve_prompt(input_payload, parameters)
    if not prompt:
        raise OllamaAdapterError("Prompt is required for Ollama text generation.", 422)

    request_body = {
        "model": entry.get("runtime_model") or model_id,
        "prompt": prompt,
        "stream": False,
        "options": _ollama_options(parameters),
    }

    # OLLAMA_BASE_URL overrides registry endpoints for deployments where the
    # Ollama runtime is not the compose-internal service (e.g. host systemd).
    endpoint = str(entry["endpoint"])
    base_override = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
    if base_override:
        endpoint = f"{base_override}/api/generate"

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=False) as client:
            response = await client.post(endpoint, json=request_body, headers={"X-Request-ID": request_id})
    except httpx.TimeoutException as exc:
        raise OllamaAdapterError(f"Request timed out after {int(timeout_seconds)} seconds.", 504) from exc
    except httpx.RequestError as exc:
        raise OllamaAdapterError(f"Could not reach Ollama runtime: {exc}", 503) from exc

    if response.status_code >= 400:
        raise OllamaAdapterError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}", response.status_code)

    try:
        body = response.json()
    except ValueError as exc:
        raise OllamaAdapterError("Ollama returned invalid JSON.") from exc

    output_text = body.get("response")
    if not isinstance(output_text, str) or not output_text.strip():
        thinking = body.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            output_text = thinking
        elif body.get("done_reason") == "length":
            raise OllamaAdapterError(
                "Token limit reached before the model produced a final answer. "
                "Reasoning models spend tokens thinking first - increase max_tokens.",
                422,
            )
        else:
            raise OllamaAdapterError("Ollama response did not include generated text.")

    return {
        "success": True,
        "status": "completed",
        "model": model_id,
        "model_id": model_id,
        "output_text": output_text,
        "parameters_used": request_body["options"],
        "usage": {
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "total_duration": body.get("total_duration"),
        },
        "metadata": {
            "provider": "ollama",
            "runtime_model": request_body["model"],
        },
    }
