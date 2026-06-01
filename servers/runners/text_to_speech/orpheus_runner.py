from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class OrpheusTTSRunner:
    """Runner for OrpheusTTS/Orpheus-3b-0.1-ft and canopylabs/orpheus-tts-0.1-finetune-prod."""

    def __init__(self, variant: str = "orpheus-3b"):
        self.variant = variant
        if "canopylabs" in variant or "prod" in variant:
            self.model_id = "canopylabs/orpheus-tts-0.1-finetune-prod"
            self.hf_dir = "canopylabs--orpheus-tts-0.1-finetune-prod"
        else:
            self.model_id = "OrpheusTTS/Orpheus-3b-0.1-ft"
            self.hf_dir = "OrpheusTTS--Orpheus-3b-0.1-ft"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / self.hf_dir
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.tokenizer = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForCausalLM.from_pretrained(source, torch_dtype="auto")

    def generate(self, *, text: str, output_path: str, **kwargs) -> dict:
        self.load()
        import torch

        prompt = f"<|text|>{text}<|audio|>"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.7, do_sample=True)

        audio_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        audio = self._decode_audio_tokens(audio_tokens)

        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}

    def _decode_audio_tokens(self, tokens) -> np.ndarray:
        token_values = tokens.cpu().numpy().astype(np.float32)
        audio = (token_values - token_values.mean()) / (token_values.std() + 1e-8)
        return audio
