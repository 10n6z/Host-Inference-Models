from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class KokoroONNXRunner:
    """Runner for onnx-community/Kokoro-82M-v1.0-ONNX using onnxruntime."""

    def __init__(self):
        self.model_dir = Path(
            os.getenv("HF_MODELS_ROOT", "models/hf")
        ) / "onnx-community--Kokoro-82M-v1.0-ONNX"
        self.session = None
        self.tokenizer = None
        self.voices = None
        self.sample_rate = 24000

    def load(self):
        if self.session is not None:
            return

        import json
        import onnxruntime as ort

        onnx_path = self.model_dir / "onnx" / "model_quantized.onnx"
        if not onnx_path.exists():
            onnx_path = self.model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"No ONNX model found in {self.model_dir / 'onnx'}")

        self.session = ort.InferenceSession(str(onnx_path))

        voices_path = self.model_dir / "voices-v1.0.bin"
        if voices_path.exists():
            npz = np.load(str(voices_path), allow_pickle=True)
            self.voices = {k: npz[k] for k in npz.files}

        tokenizer_path = self.model_dir / "tokenizer.json"
        if tokenizer_path.exists():
            with open(tokenizer_path) as f:
                tok_data = json.load(f)
            self.tokenizer = tok_data.get("model", {}).get("vocab", {})

    def _get_voice_style(self, voice_name: str) -> np.ndarray:
        if self.voices is None:
            raise RuntimeError("Voice embeddings not loaded")
        if voice_name not in self.voices:
            voice_name = list(self.voices.keys())[0]
        embedding = self.voices[voice_name]
        # Shape is [N, 1, 256]. Use first frame as style vector.
        style = embedding[0, 0, :]  # [256]
        return style.reshape(1, 256).astype(np.float32)

    def _phonemize(self, text: str) -> str:
        """Convert text to phonemes using kokoro's pipeline."""
        try:
            from kokoro import KPipeline
            pipeline = KPipeline(lang_code="a")
            result = pipeline.g2p(text)
            if isinstance(result, tuple):
                return result[0]
            return str(result)
        except Exception:
            pass
        return text

    def _tokenize(self, text: str) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        phonemes = self._phonemize(text)
        ids = []
        for ch in phonemes:
            if ch in self.tokenizer:
                ids.append(self.tokenizer[ch])
            elif ch == " ":
                ids.append(self.tokenizer.get(" ", 16))
        if not ids:
            ids = [0]
        return np.array([ids], dtype=np.int64)

    def list_voices(self) -> list[str]:
        self.load()
        if self.voices is None:
            return []
        return sorted(self.voices.keys())

    def generate(self, *, text: str, output_path: str, voice: str = "af_heart", speed: float = 1.0, **kwargs) -> dict:
        self.load()

        tokens = self._tokenize(text)
        style = self._get_voice_style(voice)
        speed_arr = np.array([speed], dtype=np.float32)

        outputs = self.session.run(None, {
            "input_ids": tokens,
            "style": style,
            "speed": speed_arr,
        })
        audio = outputs[0].squeeze()

        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "voice": voice,
                "speed": speed,
            },
            "available_voices": self.list_voices(),
        }
