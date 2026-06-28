from __future__ import annotations

import os

import numpy as np
import soundfile as sf

TEXT_MAX_LENGTH = 300

# Emotion / sound event tags the turbo model understands inline in `text`.
EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]",
]


class ChatterboxTurboRunner:
    """Local runner for ResembleAI/chatterbox-turbo.

    Supports inline event tags (e.g. "[chuckle]", "[sigh]") in the text input.
    """

    def __init__(self):
        self.model_id = "ResembleAI/chatterbox-turbo"
        self.model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self.model = ChatterboxTurboTTS.from_pretrained(device=os.getenv("DEVICE", "cpu"))
        self.sample_rate = self.model.sr

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        audio_prompt_path: str | None = None,
        temperature: float = 0.8,
        seed_num: int = 0,
        min_p: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
        norm_loudness: bool = True,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        if seed_num and seed_num > 0:
            torch.manual_seed(seed_num)
            np.random.seed(seed_num)

        with torch.no_grad():
            wav = self.model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path or None,
                temperature=float(temperature),
                min_p=float(min_p),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                norm_loudness=bool(norm_loudness),
            )

        audio = wav.cpu().numpy().squeeze()
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)

        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "audio_prompt_path": audio_prompt_path or None,
                "temperature": temperature,
                "seed_num": seed_num,
                "min_p": min_p,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "norm_loudness": norm_loudness,
            },
        }
