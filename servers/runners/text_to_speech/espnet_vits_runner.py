from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class ESPnetVITSRunner:
    """Runner for espnet/kan-bayashi_ljspeech_vits."""

    def __init__(self):
        self.model_id = "espnet/kan-bayashi_ljspeech_vits"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "espnet--kan-bayashi_ljspeech_vits"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 22050

    def load(self):
        if self.model is not None:
            return

        from espnet2.bin.tts_inference import Text2Speech

        source = str(self.local_path) if self.local_path else self.model_id
        self.model = Text2Speech.from_pretrained(model_tag=self.model_id)
        self.sample_rate = 22050

    def generate(self, *, text: str, output_path: str, **kwargs) -> dict:
        self.load()
        import torch

        with torch.no_grad():
            output = self.model(text)

        audio = output["wav"].cpu().numpy()
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}
