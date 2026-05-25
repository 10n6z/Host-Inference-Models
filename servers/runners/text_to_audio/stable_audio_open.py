from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class StableAudioOpenRunner:
    def __init__(self):
        self.model_path = os.getenv(
            "STABLE_AUDIO_MODEL_PATH",
            "/home/long/local-ai/models/text-to-audio/stable-audio-open-1.0",
        )
        self.pipe = None
        self.device = "cpu"
        self.sample_rate_fallback = 44100

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Stable Audio Open model folder not found: {self.model_path}")

        import torch
        from diffusers import StableAudioPipeline

        pipe = StableAudioPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            self.device = "cuda"

        self.pipe = pipe
        return self.pipe

    def generate(
        self,
        *,
        prompt: str,
        output_path: str,
        negative_prompt: str | None = None,
        duration_seconds: float = 10.0,
        steps: int = 100,
        guidance_scale: float = 7.0,
        seed: int | None = None,
        format: str = "wav",
        random_seed: bool = True,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if format.lower() != "wav":
            raise ValueError("Stable Audio Open runner supports format=wav only.")

        pipe = self.load()

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(self.device).manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            audio_end_in_s=float(duration_seconds),
            generator=generator,
            num_waveforms_per_prompt=1,
        )

        audio = self._to_soundfile_array(result.audios[0])
        sample_rate = self._sample_rate(pipe)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), audio, sample_rate)

        duration = float(audio.shape[0] / sample_rate) if sample_rate > 0 else None
        channels = int(audio.shape[1]) if audio.ndim == 2 else 1
        return {
            "output_path": str(output),
            "sample_rate": sample_rate,
            "audio_duration_seconds": duration,
            "channels": channels,
        }

    def _to_soundfile_array(self, audio: Any) -> np.ndarray:
        try:
            import torch
        except Exception:
            torch = None

        if torch is not None and isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu().numpy()
        else:
            audio = np.asarray(audio)

        if audio.ndim == 3:
            audio = audio[0]

        if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T

        return np.asarray(audio, dtype=np.float32)

    def _sample_rate(self, pipe: Any) -> int:
        vae = getattr(pipe, "vae", None)
        sample_rate = getattr(vae, "sampling_rate", None)
        if sample_rate is None:
            sample_rate = getattr(pipe, "sampling_rate", None)
        return int(sample_rate or self.sample_rate_fallback)
