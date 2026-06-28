from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class F5TTSRunner:
    """Runner for SWivid/F5-TTS."""

    def __init__(self):
        self.model_id = "SWivid/F5-TTS"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "SWivid--F5-TTS"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 24000
        self._ref_file = None

    def _ensure_reference_audio(self) -> str:
        """Create a short silence reference if no reference provided."""
        if self._ref_file and os.path.exists(self._ref_file):
            return self._ref_file
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        silence = np.zeros(24000, dtype=np.float32)
        sf.write(path, silence, 24000)
        self._ref_file = path
        return path

    def load(self):
        if self.model is not None:
            return

        from f5_tts.api import F5TTS

        self.model = F5TTS(model="F5TTS_v1_Base", ckpt_file="", device=os.getenv("DEVICE", "cpu"))
        self.sample_rate = 24000

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        ref_file: str = "",
        ref_text: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> dict:
        self.load()

        actual_ref = ref_file if ref_file and os.path.exists(ref_file) else self._ensure_reference_audio()
        actual_ref_text = ref_text if ref_text else "silence"

        audio, sr, _ = self.model.infer(
            ref_file=actual_ref,
            ref_text=actual_ref_text,
            gen_text=text,
            speed=speed,
        )
        self.sample_rate = sr if sr else self.sample_rate
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "ref_file": ref_file,
                "ref_text": ref_text,
                "speed": speed,
            },
        }
