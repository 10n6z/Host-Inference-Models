from __future__ import annotations

import os

import numpy as np
import soundfile as sf

SUPPORTED_LANGUAGES = [
    "en", "ar", "da", "de", "el", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
]

TEXT_MAX_LENGTH = 300


class ChatterboxMultilingualRunner:
    """Local runner for Chatterbox Multilingual V3."""

    def __init__(self):
        self.model_id = "ResembleAI/chatterbox"
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_pretrained(
            device=os.getenv("DEVICE", "cpu"),
            t3_model="v3",
        )
        self.sample_rate = self.model.sr

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        language_id: str = "en",
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        seed_num: int = 0,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        if language_id not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        if seed_num and seed_num > 0:
            torch.manual_seed(seed_num)
            np.random.seed(seed_num)

        with torch.no_grad():
            wav = self.model.generate(
                text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path or None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        audio = wav.cpu().numpy().squeeze()
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)

        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "language_id": language_id,
                "audio_prompt_path": audio_prompt_path or None,
                "exaggeration": exaggeration,
                "temperature": temperature,
                "seed_num": seed_num,
                "cfg_weight": cfg_weight,
            },
        }
