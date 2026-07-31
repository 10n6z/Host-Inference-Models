"""Metadata-only payload shared by every commercial planner adapter.

Only compact, non-identifying task metadata crosses the commercial-provider
boundary -- never image bytes, document content, crops, OCR text, detection
evidence, or raw requested labels (only their count).
"""

from __future__ import annotations

from pydantic import BaseModel


class PlanningMetadata(BaseModel):
    prompt_classifier_hints: list[str]
    image_count: int
    requested_domain: str
    requested_tasks: list[str]
    ocr_mode: str
    ocr_language: str
    detector_mode: str
    requested_label_count: int


class JobFixture(BaseModel):
    """Full job context available control-plane-side, used only to derive
    the compact PlanningMetadata below -- never sent to a commercial provider
    as-is."""

    prompt: str
    image_count: int
    requested_domain: str
    requested_tasks: list[str]
    ocr_mode: str
    ocr_language: str
    detector_mode: str
    requested_labels: list[str]


_CLASSIFIER_KEYWORDS = {
    "compare": "comparison",
    "diff": "comparison",
    "safety": "safety",
    "hazard": "safety",
    "ppe": "safety",
    "count": "counting",
    "bug": "ui_qa",
    "layout": "ui_qa",
    "read": "ocr",
    "extract": "ocr",
    "label": "ocr",
}


def _classifier_hints(prompt: str) -> list[str]:
    lowered = prompt.lower()
    return sorted({hint for word, hint in _CLASSIFIER_KEYWORDS.items() if word in lowered})


def build_planning_metadata(job: JobFixture) -> PlanningMetadata:
    return PlanningMetadata(
        prompt_classifier_hints=_classifier_hints(job.prompt),
        image_count=job.image_count,
        requested_domain=job.requested_domain,
        requested_tasks=job.requested_tasks,
        ocr_mode=job.ocr_mode,
        ocr_language=job.ocr_language,
        detector_mode=job.detector_mode,
        requested_label_count=len(job.requested_labels),
    )
