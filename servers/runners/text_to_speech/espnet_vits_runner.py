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

        self.model = Text2Speech.from_pretrained(model_tag=self.model_id)
        model_sr = getattr(self.model, "fs", None)
        if model_sr:
            self.sample_rate = int(model_sr)

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        alpha: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        decode_conf = {
            "alpha": alpha,
            "noise_scale": noise_scale,
            "noise_scale_dur": noise_scale_dur,
        }

        with torch.no_grad():
            output = self.model(text, decode_conf=decode_conf)

        audio = output["wav"].cpu().numpy()
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "alpha": alpha,
                "noise_scale": noise_scale,
                "noise_scale_dur": noise_scale_dur,
            },
        }
