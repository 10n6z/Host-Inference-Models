from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class KittenTTSRunner:
    """Runner for KittenML/kitten-tts-nano-0.2."""

    def __init__(self):
        self.model_id = "KittenML/kitten-tts-nano-0.2"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "KittenML--kitten-tts-nano-0.2"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True, torch_dtype="auto")

    def generate(self, *, text: str, output_path: str, **kwargs) -> dict:
        self.load()
        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2048)

        audio_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        audio = audio_tokens.float().cpu().numpy()
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)

        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}
