from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class MeloTTSRunner:
    """Runner for myshell-ai/MeloTTS-English."""

    def __init__(self):
        self.model_id = "myshell-ai/MeloTTS-English"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "myshell-ai--MeloTTS-English"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 44100

    def load(self):
        if self.model is not None:
            return

        from melo.api import TTS

        self.model = TTS(language="EN", device="cpu")
        self.sample_rate = self.model.hps.data.sampling_rate

    def generate(self, *, text: str, output_path: str, speed: float = 1.0, **kwargs) -> dict:
        self.load()

        speaker_ids = self.model.hps.data.spk2id
        speaker_id = list(speaker_ids.values())[0]
        self.model.tts_to_file(text, speaker_id, output_path, speed=speed)

        info = sf.info(output_path)
        duration = float(info.frames / info.samplerate)
        return {"output_path": output_path, "sample_rate": info.samplerate, "duration_seconds": duration}
