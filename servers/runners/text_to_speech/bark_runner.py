from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write as write_wav

# Bark checkpoints are pickled; torch>=2.6 defaults torch.load(weights_only=True),
# which rejects them. Restore the permissive default for these trusted weights.
import torch

_ORIG_TORCH_LOAD = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _patched_torch_load


class BarkRunner:
    """Runner backed by the official suno-ai/bark package.

    Mirrors the upstream demo / README: ``generate_audio(text, history_prompt=...)``
    with the model's own pipeline (BertTokenizer + EnCodec) on GPU when available.
    ``voice_preset`` maps to ``history_prompt`` (empty/falsy = Unconditional).
    Both bark-small and bark-full endpoints serve the full official models.
    """

    def __init__(self, variant: str = "full"):
        self.variant = variant
        self.sample_rate = 24000
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from bark import SAMPLE_RATE, preload_models

        preload_models()
        self.sample_rate = SAMPLE_RATE
        self._loaded = True

    @staticmethod
    def _seed_all(seed: int) -> None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        voice_preset: str = "v2/en_speaker_6",
        do_sample: bool = True,
        temperature: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> dict:
        self.load()
        from bark import generate_audio

        if seed is not None:
            self._seed_all(int(seed))

        history_prompt = voice_preset or None  # empty/falsy = Unconditional (random voice)
        temp = 1.0 if temperature is None else float(temperature)

        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=temp,
            waveform_temp=temp,
        )
        audio_array = np.asarray(audio_array, dtype=np.float32)
        write_wav(output_path, self.sample_rate, audio_array)
        duration = float(len(audio_array) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "voice_preset": voice_preset,
                "do_sample": do_sample,
                "temperature": temp,
                "seed": int(seed) if seed is not None else None,
            },
        }
