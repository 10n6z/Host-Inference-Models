from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import soundfile as sf


class IndexTTS2Runner:
    def __init__(self):
        self.model_path = Path(
            os.getenv(
                "INDEXTTS2_MODEL_PATH",
                "/home/long/local-ai/models/text-to-speech/indextts-2",
            )
        )
        self.cfg_path = Path(os.getenv("INDEXTTS2_CONFIG_PATH", str(self.model_path / "config.yaml")))
        self.fp16 = os.getenv("INDEXTTS2_FP16", "false").lower() == "true"
        self.use_cuda_kernel = os.getenv("INDEXTTS2_USE_CUDA_KERNEL", "false").lower() == "true"
        self.use_deepspeed = os.getenv("INDEXTTS2_USE_DEEPSPEED", "false").lower() == "true"
        self.model = None

    def load(self):
        if self.model is not None:
            return self.model

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"IndexTTS-2 model folder not found: {self.model_path}")
        if not self.cfg_path.is_file():
            raise FileNotFoundError(f"IndexTTS-2 config file not found: {self.cfg_path}")

        try:
            from indextts.infer_v2 import IndexTTS2
        except Exception as exc:
            raise FileNotFoundError(
                "IndexTTS2 runtime dependencies not available. "
                "Install index-tts runtime before using this runner."
            ) from exc

        self.model = IndexTTS2(
            cfg_path=str(self.cfg_path),
            model_dir=str(self.model_path),
            use_fp16=self.fp16,
            use_cuda_kernel=self.use_cuda_kernel,
            use_deepspeed=self.use_deepspeed,
        )
        return self.model

    @staticmethod
    def _resolve_reference_audio_path(reference_audio_id: str | None) -> Path:
        if reference_audio_id:
            candidate = Path(reference_audio_id)
            if candidate.is_file():
                return candidate

            ref_root = Path(os.getenv("TTS_REFERENCE_AUDIO_ROOT", "/home/long/local-ai/references"))
            rooted = ref_root / reference_audio_id
            if rooted.is_file():
                return rooted

        default_path = Path(os.getenv("INDEXTTS2_DEFAULT_REFERENCE_AUDIO", ""))
        if default_path.is_file():
            return default_path

        raise FileNotFoundError(
            "IndexTTS2 requires reference audio. "
            "Provide reference_audio_id path or set INDEXTTS2_DEFAULT_REFERENCE_AUDIO."
        )

    @staticmethod
    def _maybe_read_duration(path: Path) -> float | None:
        try:
            info = sf.info(str(path))
            if info.samplerate > 0:
                return float(info.frames / info.samplerate)
        except Exception:
            return None
        return None

    @staticmethod
    def _call_with_supported_kwargs(func, **kwargs):
        sig = inspect.signature(func)
        call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
        return func(**call_kwargs)

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        speaker: str = "default",
        speed: float = 1.0,
        format: str = "wav",
        reference_audio_id: str | None = None,
        emotion: str | None = None,
        duration_control: float | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if format.lower() != "wav":
            raise ValueError("IndexTTS2 runner currently supports format=wav only.")

        model = self.load()
        ref_audio = self._resolve_reference_audio_path(reference_audio_id)

        # IndexTTS2 API uses spk_audio_prompt + optional emotion controls.
        emo_alpha = None
        if duration_control is not None:
            # Keep schema compatibility: map duration_control into token/tempo scaling proxy.
            # Actual IndexTTS2 duration token control is model-specific; this is soft mapping.
            emo_alpha = max(0.0, min(1.0, float(duration_control) / 2.0))

        self._call_with_supported_kwargs(
            model.infer,
            spk_audio_prompt=str(ref_audio),
            text=text,
            output_path=output_path,
            emo_audio_prompt=str(ref_audio),
            emo_alpha=emo_alpha,
            use_emo_text=bool(emotion),
            emo_text=emotion,
            use_random=False,
            verbose=False,
        )

        if not Path(output_path).is_file():
            raise RuntimeError("IndexTTS2 did not write output audio file.")

        info = sf.info(output_path)
        duration_seconds = self._maybe_read_duration(Path(output_path))
        return {
            "output_path": output_path,
            "sample_rate": int(info.samplerate),
            "duration_seconds": duration_seconds,
            "speaker": speaker,
            "speed": speed,
        }
