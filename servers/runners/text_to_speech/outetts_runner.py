from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class OuteTTSRunner:
    """Runner for OuteAI/OuteTTS-0.2-500M and OuteAI/OuteTTS-0.3-1B."""

    def __init__(self, variant: str = "0.2-500M"):
        self.variant = variant
        if "0.3" in variant:
            self.model_id = "OuteAI/OuteTTS-0.3-1B"
            self.hf_dir = "OuteAI--OuteTTS-0.3-1B"
        else:
            self.model_id = "OuteAI/OuteTTS-0.2-500M"
            self.hf_dir = "OuteAI--OuteTTS-0.2-500M"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / self.hf_dir
        self.local_path = local_path if local_path.is_dir() else None
        self.interface = None
        self.sample_rate = 24000

    def load(self):
        if self.interface is not None:
            return

        import outetts

        config = outetts.HFModelConfig_v2(
            model_path=self.model_id,
            tokenizer_path=self.model_id,
        )
        self.interface = outetts.InterfaceHF(model_version="0.3" if "0.3" in self.variant else "0.2", cfg=config)

    def generate(self, *, text: str, output_path: str, **kwargs) -> dict:
        self.load()

        output = self.interface.generate(text=text, temperature=0.1, repetition_penalty=1.1, max_length=4096)
        output.save(output_path)

        info = sf.info(output_path)
        duration = float(info.frames / info.samplerate)
        return {"output_path": output_path, "sample_rate": info.samplerate, "duration_seconds": duration}
