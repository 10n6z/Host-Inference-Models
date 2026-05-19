from __future__ import annotations

from typing import Any


class FishSpeechRunner:
    """
    Placeholder runner for fishaudio/fish-speech-1.5.

    This server exposes strict API contracts first.
    Actual runtime integration can replace this class without changing route schemas.
    """
    available = False
    unavailable_reason = "Runner integration not wired yet for fishaudio/fish-speech-1.5."

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        language: str = "en",
        voice: str = "default",
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int | None = None,
        reference_audio_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise FileNotFoundError(
            "Model folder not found or runner not wired: fishaudio/fish-speech-1.5. "
            "Implement FishSpeechRunner.generate with local inference integration."
        )
