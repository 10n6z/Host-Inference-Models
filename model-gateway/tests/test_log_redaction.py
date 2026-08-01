from __future__ import annotations

import ast
import re
from pathlib import Path

MAIN_PY = Path(__file__).resolve().parents[1] / "main.py"

# Anything that looks like it could carry request/response body, image,
# document, OCR, provider-content, or credential data. Argument identifiers
# are matched by their full dotted name (e.g. "upstream.status_code" is
# fine; "upstream_body" or "raw_body" is not).
PROHIBITED_ARG_PATTERN = re.compile(
    r"body|payload|image|document|ocr|credential|api_key|apikey|password|secret",
    re.IGNORECASE,
)

# `status_code` legitimately contains "code" but not body content; allow it
# explicitly since it would otherwise be fine anyway -- kept for clarity.
ALLOWED_EXACT = {"upstream.status_code"}


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _logger_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "logger"
            and func.attr in {"info", "warning", "error", "debug", "critical", "exception"}
        ):
            yield node


def test_no_logger_call_references_body_image_document_ocr_or_credential_variables():
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"), filename=str(MAIN_PY))
    violations = []
    calls = list(_logger_calls(tree))
    assert calls, "expected at least one logger.* call in model-gateway/main.py"
    for call in calls:
        # First arg is the format string; the rest are the interpolated values.
        for arg in call.args[1:]:
            name = _dotted_name(arg)
            if name is None:
                continue
            if name in ALLOWED_EXACT:
                continue
            if PROHIBITED_ARG_PATTERN.search(name):
                violations.append((call.lineno, name))
    assert violations == [], (
        "logger.* call(s) reference variables that look like they could carry "
        f"request/response body, image, document, OCR, provider, or credential data: {violations}"
    )
