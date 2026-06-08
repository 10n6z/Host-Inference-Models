from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class XTTSRunner:
    """Runner for coqui/XTTS-v2."""

    def __init__(self):
        self.model_id = "coqui/XTTS-v2"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "coqui--XTTS-v2"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.sample_rate = 24000
        self._ref_file = None

    def _ensure_reference_audio(self) -> str:
        if self._ref_file and os.path.exists(self._ref_file):
            return self._ref_file
        # Use bundled sample from XTTS-v2 repo (requires >3s reference)
        model_root = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "coqui--XTTS-v2"
        sample = model_root / "samples" / "en_sample.wav"
        if sample.exists():
            self._ref_file = str(sample)
            return self._ref_file
        # Fallback: generate 4s silence
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        silence = np.zeros(24000 * 4, dtype=np.float32)
        sf.write(path, silence, 24000)
        self._ref_file = path
        return path

    def load(self):
        if self.model is not None:
            return

        import torch
        # TTS 0.22.0 predates PyTorch 2.6's weights_only=True default
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        source = str(self.local_path) if self.local_path else self.model_id
        config = XttsConfig()
        config.load_json(os.path.join(source, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=source)

        torch.load = _orig_load

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        language: str = "en",
        speaker_wav: str = "",
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        ref = speaker_wav if speaker_wav and os.path.exists(speaker_wav) else self._ensure_reference_audio()

        with torch.no_grad():
            outputs = self.model.synthesize(
                text,
                config=self.model.config,
                speaker_wav=ref,
                language=language,
            )

        audio = np.array(outputs["wav"])
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {"output_path": output_path, "sample_rate": self.sample_rate, "duration_seconds": duration}
