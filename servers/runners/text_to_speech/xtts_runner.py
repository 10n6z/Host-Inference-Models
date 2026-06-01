from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class XTTSRunner:
    """Runner for coqui/XTTS-v2."""

    def __init__(self):
        self.model_id = "coqui/XTTS-v2"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "coqui--XTTS-v2"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        source = str(self.local_path) if self.local_path else self.model_id
        config = XttsConfig()
        config.load_json(os.path.join(source, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=source)

    def generate(self, *, text: str, output_path: str, language: str = "en", **kwargs) -> dict:
        self.load()
        import torch

        with torch.no_grad():
            outputs = self.model.synthesize(text, config=self.model.config, language=language)

        audio = np.array(outputs["wav"])
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}
