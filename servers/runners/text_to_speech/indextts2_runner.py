from __future__ import annotations

from typing import Any


class IndexTTS2Runner:
    """
    Placeholder runner for IndexTeam/IndexTTS-2.

    Contracts shipped now; runtime wiring can be attached without changing API surface.
    """
    available = False
    unavailable_reason = "Runner integration not wired yet for IndexTeam/IndexTTS-2."

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        speaker: str = "default",
        speed: float = 1.0,
        format: str = "wav",
        reference_audio_id: str | None = None,
        emotion: str | None = None,
        duration_control: float | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise FileNotFoundError(
            "Model folder not found or runner not wired: IndexTeam/IndexTTS-2. "
            "Implement IndexTTS2Runner.generate with local inference integration."
        )
