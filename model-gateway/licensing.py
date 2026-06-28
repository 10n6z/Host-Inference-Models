"""Central license / commercial-use metadata for registry models.

Single source of truth for the GPT-Lab Sandbox license audit. The gateway
attaches `license` and `commercial_use` to every `/models` entry so the
control-plane and UI can surface commercial eligibility without hardcoding
licenses downstream.

Resolution order for a model:
  1. Explicit `license` / `commercial_use` keys on the registry entry.
  2. `LICENSE_TABLE` lookup by model id.
  3. Prefix/family heuristic (LICENSE_PREFIXES).
  4. Conservative default: license "unknown", commercial_use False.

`commercial_use` is a tri-state string so the UI can distinguish a vetted
"yes"/"no" from an unverified "unknown":
  - "yes"     : license permits commercial use.
  - "no"      : license forbids or restricts commercial use.
  - "unknown" : not yet verified (treated as non-commercial for counts).
"""

from __future__ import annotations

from typing import Any

# License identifiers that permit commercial use (SPDX-ish, lowercased).
COMMERCIAL_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "openrail",
    "openrail++",
    "creativeml-openrail-m",
    "bigscience-openrail-m",
    "llama2",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "llama4",
    "gemma",
    "qwen",
    "tongyi-qianwen",
    "stabilityai-community",
    "flux-1-dev-non-commercial"  # placeholder, overridden to no below
}

# Licenses explicitly NON-commercial / research-only.
NON_COMMERCIAL_LICENSES = {
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "flux-1-dev-non-commercial",
    "magi-non-commercial",
    "research-only",
    "non-commercial",
}

# Per-model license assignments (verified against model cards / repos).
# Keys are registry model ids (text models keep their ollama tags).
LICENSE_TABLE: dict[str, str] = {
    # ---- image ----
    "flux-1-schnell": "apache-2.0",
    "flux-2-klein-4b": "apache-2.0",
    "qwen-image-edit-2509": "apache-2.0",
    "stable-diffusion-3.5-medium": "stabilityai-community",
    "auraflow-v0.3": "apache-2.0",
    "openflux-1": "apache-2.0",
    # ---- video ----
    "wan21-t2v-1.3b": "apache-2.0",
    "wan21-t2v-14b": "apache-2.0",
    "magi-1-4.5b": "apache-2.0",
    "framepack-i2v": "apache-2.0",
    "framepack-t2v": "apache-2.0",
    "skyreels-v2-df-1.3b-540p": "apache-2.0",
    "cogvideox-2b": "apache-2.0",
    "ltx-video-2b": "openrail++",
    # ---- audio ----
    "kokoro-82m": "apache-2.0",
    "kokoro-82m-onnx": "apache-2.0",
    "melotts": "mit",
    "outetts": "cc-by-nc-4.0",
    "stable-audio-open-1.0": "stabilityai-community",
    "mms-tts": "cc-by-nc-4.0",
    "speecht5-tts": "mit",
    "bark-small": "mit",
    "bark-full": "mit",
    "espnet-vits": "apache-2.0",
    "chatterbox": "mit",
    "chatterbox-multilingual": "mit",
    "chatterbox-turbo": "mit",
    "f5-tts": "cc-by-nc-4.0",
    "e2-tts": "cc-by-nc-4.0",
    "kitten-tts": "apache-2.0",
    "xtts-v2": "cc-by-nc-4.0",
    "csm-1b": "apache-2.0",
}

# Family/prefix heuristics for text (ollama-tag) models. Checked in order.
LICENSE_PREFIXES: list[tuple[str, str]] = [
    ("deepseek-coder", "mit"),
    ("deepseek-r1", "mit"),
    ("deepseek-v2", "deepseek"),
    ("gemma", "gemma"),
    ("gpt-oss", "apache-2.0"),
    ("second_constantine/gpt-oss", "apache-2.0"),
    ("svjack/gpt-oss", "apache-2.0"),
    ("llama-poro-2", "llama3.3"),
    ("hf.co/mradermacher/llama-poro-2", "llama3.3"),
    ("smollm2", "apache-2.0"),
    ("hf.co/unsloth/smollm2", "apache-2.0"),
    ("llama2-uncensored", "llama2"),
    ("llama2", "llama2"),
    ("llama3.1", "llama3.1"),
    ("llama3.2", "llama3.2"),
    ("llama3.3", "llama3.3"),
    ("llama4", "llama4"),
    ("llama3", "llama3"),
    ("llava", "apache-2.0"),
    ("mistral", "apache-2.0"),
    ("phi4", "mit"),
    ("qwen2.5-coder", "apache-2.0"),
    ("qwen3-coder", "apache-2.0"),
    ("qwen3.5", "qwen"),
    ("qwen3.6", "qwen"),
    ("qwen3", "apache-2.0"),
    ("qwen", "qwen"),
    ("starcoder2", "bigcode-openrail-m"),
]

# Extra commercial-friendly licenses surfaced only via heuristics.
_COMMERCIAL_EXTRA = {"deepseek", "bigcode-openrail-m"}


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _commercial_status(license_id: str) -> str:
    norm = _normalize(license_id)
    if not norm or norm == "unknown":
        return "unknown"
    if norm in NON_COMMERCIAL_LICENSES:
        return "no"
    if norm in COMMERCIAL_LICENSES or norm in _COMMERCIAL_EXTRA:
        return "yes"
    return "unknown"


def resolve_license(model_id: str, entry: dict[str, Any]) -> dict[str, str]:
    """Return {"license", "commercial_use"} for a registry model entry."""
    explicit_license = entry.get("license")
    if explicit_license:
        license_id = str(explicit_license)
        explicit_commercial = entry.get("commercial_use")
        commercial = (
            _normalize(explicit_commercial)
            if explicit_commercial is not None
            else _commercial_status(license_id)
        )
        if commercial not in {"yes", "no", "unknown"}:
            commercial = _commercial_status(license_id)
        return {"license": license_id, "commercial_use": commercial}

    key = _normalize(model_id)
    if key in {k.lower() for k in LICENSE_TABLE}:
        # Resolve case-insensitively while keeping the table's canonical id.
        for table_id, lic in LICENSE_TABLE.items():
            if table_id.lower() == key:
                return {"license": lic, "commercial_use": _commercial_status(lic)}

    for prefix, lic in LICENSE_PREFIXES:
        if key.startswith(prefix):
            return {"license": lic, "commercial_use": _commercial_status(lic)}

    return {"license": "unknown", "commercial_use": "unknown"}
