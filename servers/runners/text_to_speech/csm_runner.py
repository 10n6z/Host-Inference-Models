from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class CSMRunner:
    """Runner for sesame/csm-1b."""

    def __init__(self):
        self.model_id = "sesame/csm-1b"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "sesame--csm-1b"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)

    def generate(self, *, text: str, output_path: str, **kwargs) -> dict:
        self.load()
        import torch

        with torch.no_grad():
            audio = self.model.generate(text=text)

        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio = np.squeeze(audio).astype(np.float32)

        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}
