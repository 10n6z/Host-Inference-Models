from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf

ASSETS_DIR = Path(__file__).resolve().parents[3] / "services" / "audio-cosyvoice" / "assets"
DEFAULT_REF_WAV = os.getenv("COSYVOICE2_REF_WAV", str(ASSETS_DIR / "ref_en.wav"))
DEFAULT_REF_TXT = os.getenv("COSYVOICE2_REF_TXT", str(ASSETS_DIR / "ref_en.txt"))


class CosyVoice2Runner:
    def __init__(self):
        self.model_path = os.getenv("COSYVOICE2_MODEL_PATH", "/gpt-lab/long/models/text-to-speech/cosyvoice2-0.5b")
        self.model = None
        self.sample_rate = 24000
        self._prompt_speech = None
        self._prompt_text = None

    def load(self):
        if self.model is not None:
            return
        import sys

        matcha = "/app/CosyVoice/third_party/Matcha-TTS"
        if matcha not in sys.path:
            sys.path.append(matcha)
        from cosyvoice.cli.cosyvoice import CosyVoice2

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"CosyVoice2 model folder not found: {self.model_path}")
        self.model = CosyVoice2(self.model_path, load_jit=False, load_trt=False, fp16=False)
        model_sr = getattr(self.model, "sample_rate", None)
        if model_sr:
            self.sample_rate = int(model_sr)
        self._prompt_speech = DEFAULT_REF_WAV
        with open(DEFAULT_REF_TXT, "r", encoding="utf-8") as handle:
            self._prompt_text = handle.read().strip()

    def generate(self, text: str, output_path: str, **kwargs) -> dict:
        self.load()
        outputs = []
        for item in self.model.inference_zero_shot(text, self._prompt_text, self._prompt_speech, stream=False):
            chunk = item.get("tts_speech")
            if chunk is not None:
                if hasattr(chunk, "detach"):
                    chunk = chunk.detach().cpu().numpy()
                outputs.append(np.squeeze(chunk).astype(np.float32))
        if not outputs:
            raise RuntimeError("CosyVoice2 generated no audio.")
        audio = np.concatenate(outputs)
        sf.write(output_path, audio, self.sample_rate)
        return {"sample_rate": self.sample_rate, "duration_seconds": float(len(audio) / self.sample_rate)}
