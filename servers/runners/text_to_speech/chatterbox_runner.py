from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class ChatterboxRunner:
    """Runner for ResembleAI/chatterbox."""

    def __init__(self):
        self.model_id = "ResembleAI/chatterbox"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "ResembleAI--chatterbox"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device="cpu")
        self.sample_rate = self.model.sr

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        with torch.no_grad():
            wav = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        audio = wav.cpu().numpy().squeeze()
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            },
        }
