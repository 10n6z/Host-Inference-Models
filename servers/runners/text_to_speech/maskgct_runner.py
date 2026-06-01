from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class MaskGCTRunner:
    """Runner for Amphion/MaskGCT."""

    def __init__(self):
        self.model_id = "Amphion/MaskGCT"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "Amphion--MaskGCT"
        self.local_path = local_path if local_path.is_dir() else None
        self.pipeline = None
        self.sample_rate = 24000

    def load(self):
        if self.pipeline is not None:
            return

        from transformers import pipeline

        self.pipeline = pipeline("text-to-speech", model=self.model_id, device="cpu")

    def generate(self, *, text: str, output_path: str, **kwargs) -> dict:
        self.load()

        result = self.pipeline(text)
        audio = np.array(result["audio"]).squeeze()
        sr = result.get("sampling_rate", self.sample_rate)

        sf.write(output_path, audio, sr)
        duration = float(len(audio) / sr)
        return {"output_path": output_path, "sample_rate": sr, "duration_seconds": duration}
