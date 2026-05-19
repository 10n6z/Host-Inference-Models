from __future__ import annotations

from typing import Any

import numpy as np
import soundfile as sf


class StableAudioOpenRunner:
    available = True
    unavailable_reason = None

    def __init__(self):
        self._pipe = None
        self._model_id = "stabilityai/stable-audio-open-1.0"
        self._device = "cuda"
        self._sample_rate_fallback = 44100

    def _load(self):
        if self._pipe is not None:
            return self._pipe

        try:
            import torch
            from diffusers import StableAudioPipeline
        except Exception as exc:
            raise FileNotFoundError(
                "Stable Audio runtime dependencies not available. "
                "Need diffusers StableAudioPipeline + torch."
            ) from exc

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe = StableAudioPipeline.from_pretrained(self._model_id, torch_dtype=dtype)
        pipe = pipe.to(device)

        self._device = device
        self._pipe = pipe
        return self._pipe

    def generate(
        self,
        *,
        prompt: str,
        output_path: str,
        duration_seconds: float = 10.0,
        steps: int = 50,
        guidance_scale: float = 7.0,
        seed: int | None = None,
        random_seed: bool = True,
        format: str = "wav",
        negative_prompt: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if format.lower() != "wav":
            raise ValueError("Stable Audio Open runner currently supports format=wav only.")

        pipe = self._load()

        generator = None
        if not random_seed and seed is not None:
            import torch

            generator = torch.Generator(self._device).manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            audio_end_in_s=float(duration_seconds),
            generator=generator,
            num_waveforms_per_prompt=1,
        )

        audio = result.audios[0]
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        if audio.ndim == 2:
            # Diffusers commonly returns channels-first; soundfile expects samples-first.
            audio = audio.T

        sample_rate = int(getattr(pipe, "vae").sampling_rate or self._sample_rate_fallback)
        sf.write(output_path, audio, sample_rate)

        seconds = float(len(audio) / sample_rate) if sample_rate > 0 else None
        return {
            "output_path": output_path,
            "sample_rate": sample_rate,
            "audio_duration_seconds": seconds,
            "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        }
