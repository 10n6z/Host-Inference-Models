from __future__ import annotations

from typing import Any


class CosyVoice2Runner:
    """
    Placeholder runner for FunAudioLLM/CosyVoice2-0.5B.

    This preserves endpoint contract while model-specific runtime wiring lands separately.
    """
    available = False
    unavailable_reason = "Runner integration not wired yet for FunAudioLLM/CosyVoice2-0.5B."

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        language: str = "en",
        speaker: str = "default",
        speed: float = 1.0,
        format: str = "wav",
        reference_audio_id: str | None = None,
        instruction: str | None = None,
        stream: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise FileNotFoundError(
            "Model folder not found or runner not wired: FunAudioLLM/CosyVoice2-0.5B. "
            "Implement CosyVoice2Runner.generate with local inference integration."
        )
