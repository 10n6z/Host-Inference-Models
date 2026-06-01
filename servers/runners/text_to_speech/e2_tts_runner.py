from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class E2TTSRunner:
    """Runner for SWivid/E2-TTS."""

    def __init__(self):
        self.model_id = "SWivid/E2-TTS"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "SWivid--E2-TTS"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from f5_tts.api import F5TTS

        self.model = F5TTS(model_type="E2-TTS", ckpt_file="", device="cpu")
        self.sample_rate = 24000

    def generate(self, *, text: str, output_path: str, **kwargs) -> dict:
        self.load()

        audio, sr, _ = self.model.infer(ref_file="", ref_text="", gen_text=text)
        self.sample_rate = sr if sr else self.sample_rate
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}
