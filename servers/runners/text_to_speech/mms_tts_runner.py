from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class MMSTTSRunner:
    """Runner for facebook/mms-tts-* models (VITS architecture via transformers)."""

    SUPPORTED_LANGS = {
        "eng": "facebook/mms-tts-eng",
        "deu": "facebook/mms-tts-deu",
        "fra": "facebook/mms-tts-fra",
    }

    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self.model_id = self.SUPPORTED_LANGS.get(lang, f"facebook/mms-tts-{lang}")
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / f"facebook--mms-tts-{lang}"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.tokenizer = None
        self.sample_rate = 16000

    def _has_weights(self) -> bool:
        if not self.local_path:
            return False
        for ext in ("*.safetensors", "*.bin", "*.pth"):
            if list(self.local_path.glob(ext)):
                return True
        return False

    def load(self):
        if self.model is not None:
            return

        from transformers import VitsModel, AutoTokenizer

        source = str(self.local_path) if self._has_weights() else self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = VitsModel.from_pretrained(source)
        self.sample_rate = self.model.config.sampling_rate

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        speaking_rate: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_duration: float = 0.8,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(
                **inputs,
                speaking_rate=speaking_rate,
                noise_scale=noise_scale,
                noise_scale_duration=noise_scale_duration,
            )

        waveform = output.waveform[0].cpu().numpy()
        sf.write(output_path, waveform, self.sample_rate)
        duration = float(len(waveform) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "speaking_rate": speaking_rate,
                "noise_scale": noise_scale,
                "noise_scale_duration": noise_scale_duration,
            },
        }
