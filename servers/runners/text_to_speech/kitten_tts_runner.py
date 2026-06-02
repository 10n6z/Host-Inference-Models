from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


def _setup_espeak():
    """Configure espeak-ng data path from bundled espeakng_loader."""
    try:
        from espeakng_loader import get_library_path, get_data_path
        data_path = get_data_path()
        parent = str(Path(data_path).parent) if "espeak-ng-data" in str(data_path) else str(data_path)
        os.environ.setdefault("ESPEAK_DATA_PATH", parent)
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", get_library_path())
    except ImportError:
        pass


class KittenTTSRunner:
    """Runner for KittenML/kitten-tts-nano-0.2 using kittentts package."""

    def __init__(self):
        self.model_id = "KittenML/kitten-tts-nano-0.2"
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        _setup_espeak()
        import kittentts

        self.model = kittentts.KittenTTS()

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        voice: str = "expr-voice-2-m",
        speed: float = 1.0,
        **kwargs,
    ) -> dict:
        self.load()

        audio = self.model.generate(text, voice=voice, speed=speed)
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "voice": voice,
                "speed": speed,
            },
        }
