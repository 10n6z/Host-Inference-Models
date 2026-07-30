from __future__ import annotations

import soundfile as sf


class FreeVCRunner:
    """Runner for voice_conversion_models/multilingual/vctk/freevc24 (audio-to-audio VC)."""

    def __init__(self):
        self.model_name = "voice_conversion_models/multilingual/vctk/freevc24"
        self.tts = None

    def load(self):
        if self.tts is not None:
            return

        import torch
        # TTS 0.22.0 predates PyTorch 2.6's weights_only=True default
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

        from TTS.api import TTS

        tts = TTS(self.model_name, progress_bar=False)
        if torch.cuda.is_available():
            tts = tts.to("cuda")
        self.tts = tts

        torch.load = _orig_load

    def convert(self, *, source_wav: str, target_wav: str, output_path: str) -> dict:
        self.load()
        self.tts.voice_conversion_to_file(
            source_wav=source_wav,
            target_wav=target_wav,
            file_path=output_path,
        )
        info = sf.info(output_path)
        return {
            "output_path": output_path,
            "sample_rate": int(info.samplerate),
            "duration_seconds": float(info.frames / info.samplerate),
        }
