from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class KimiAudioRunner:
    """Runner for moonshine-ai/Kimi-Audio-7B-Instruct."""

    def __init__(self):
        self.model_id = "moonshine-ai/Kimi-Audio-7B-Instruct"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "moonshine-ai--Kimi-Audio-7B-Instruct"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype="auto")

    def generate(self, *, text: str, output_path: str, **kwargs) -> dict:
        self.load()
        import torch

        prompt = f"<|audio_generation|>{text}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=4096)

        audio_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        audio = audio_tokens.float().cpu().numpy()
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)

        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}
