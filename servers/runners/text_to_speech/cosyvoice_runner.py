from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class CosyVoiceRunner:
    """Runner for FunAudioLLM/CosyVoice-300M and FunAudioLLM/CosyVoice3-0.5B.

    Uses same cosyvoice runtime as CosyVoice2 but different model paths.
    """

    def __init__(self, variant: str = "300M"):
        self.variant = variant
        if "3" in variant or "0.5B" in variant.replace("300M", ""):
            self.model_id = "FunAudioLLM/CosyVoice3-0.5B"
            self.hf_dir = "FunAudioLLM--CosyVoice3-0.5B"
        else:
            self.model_id = "FunAudioLLM/CosyVoice-300M"
            self.hf_dir = "FunAudioLLM--CosyVoice-300M"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / self.hf_dir
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 22050

    def load(self):
        if self.model is not None:
            return

        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise ImportError("CosyVoice runtime not installed.") from exc

        model_dir = str(self.local_path) if self.local_path else self.model_id
        self.model = AutoModel(model_dir=model_dir)
        model_sr = getattr(self.model, "sample_rate", None)
        if model_sr:
            self.sample_rate = int(model_sr)

    def generate(self, *, text: str, output_path: str, speaker: str = "default", **kwargs) -> dict:
        self.load()

        outputs = []
        for item in self.model.inference_sft(text, speaker):
            chunk = item.get("tts_speech") if isinstance(item, dict) else None
            if chunk is not None:
                if hasattr(chunk, "detach"):
                    chunk = chunk.detach().cpu().numpy()
                outputs.append(np.squeeze(chunk).astype(np.float32))

        if not outputs:
            raise RuntimeError(f"CosyVoice {self.variant} generated no audio.")

        audio = np.concatenate(outputs)
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}
