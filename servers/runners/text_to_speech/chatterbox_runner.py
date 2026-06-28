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

        self.model = ChatterboxTTS.from_pretrained(device=os.getenv("DEVICE", "cpu"))
        self.sample_rate = self.model.sr

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        min_p: float = 0.05,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        audio_prompt_path: str | None = None,
        seed_num: int = 0,
        norm_loudness: bool = False,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        if seed_num and seed_num > 0:
            torch.manual_seed(seed_num)
            np.random.seed(seed_num)

        with torch.no_grad():
            wav = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                audio_prompt_path=audio_prompt_path or None,
            )

        audio = wav.cpu().numpy().squeeze()

        if norm_loudness:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0.0:
                audio = audio / peak * 0.99

        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "min_p": min_p,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "audio_prompt_path": audio_prompt_path or None,
                "seed_num": seed_num,
                "norm_loudness": norm_loudness,
            },
        }
