from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class CosyVoice2Runner:
    def __init__(self):
        self.model_path = Path(
            os.getenv(
                "COSYVOICE2_MODEL_PATH",
                "/home/long/local-ai/models/text-to-speech/cosyvoice2-0.5b",
            )
        )
        self.model = None
        self.sample_rate = 22050

    def _prepare_pythonpath(self):
        # CosyVoice often expects third_party/Matcha-TTS import path.
        for candidate in (
            self.model_path / "third_party" / "Matcha-TTS",
            Path.cwd() / "third_party" / "Matcha-TTS",
        ):
            if candidate.exists():
                path_str = str(candidate)
                if path_str not in sys.path:
                    sys.path.append(path_str)

    def load(self):
        if self.model is not None:
            return self.model

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"CosyVoice2 model folder not found: {self.model_path}")

        self._prepare_pythonpath()

        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except Exception as exc:
            raise FileNotFoundError(
                "CosyVoice2 runtime dependencies not available. "
                "Install CosyVoice runtime before using this runner."
            ) from exc

        self.model = AutoModel(model_dir=str(self.model_path))
        model_sr = getattr(self.model, "sample_rate", None)
        if model_sr:
            self.sample_rate = int(model_sr)
        return self.model

    @staticmethod
    def _resolve_reference_audio_path(reference_audio_id: str) -> Path:
        candidate = Path(reference_audio_id)
        if candidate.is_file():
            return candidate

        ref_root = Path(os.getenv("TTS_REFERENCE_AUDIO_ROOT", "/home/long/local-ai/references"))
        rooted = ref_root / reference_audio_id
        if rooted.is_file():
            return rooted

        raise FileNotFoundError(
            "reference_audio_id must point to an existing file path "
            f"or file under TTS_REFERENCE_AUDIO_ROOT ({ref_root})."
        )

    @staticmethod
    def _call_with_supported_kwargs(func, *args, **kwargs):
        sig = inspect.signature(func)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(*args, **supported)

    @staticmethod
    def _to_np_audio(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.ndim > 1:
            value = np.squeeze(value)
        return value.astype(np.float32, copy=False)

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        language: str = "en",
        speaker: str = "default",
        speed: float = 1.0,
        format: str = "wav",
        reference_audio_id: str | None = None,
        instruction: str | None = None,
        stream: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if format.lower() != "wav":
            raise ValueError("CosyVoice2 runner currently supports format=wav only.")

        model = self.load()
        outputs = []

        if reference_audio_id:
            reference_path = self._resolve_reference_audio_path(reference_audio_id)
            prompt_text = ""
            if isinstance(parameters, dict):
                prompt_text = str(parameters.get("referenceText", "") or "")

            if instruction:
                generator = self._call_with_supported_kwargs(
                    model.inference_instruct2,
                    text,
                    instruction,
                    str(reference_path),
                    stream=stream,
                    speed=float(speed),
                )
            else:
                generator = self._call_with_supported_kwargs(
                    model.inference_zero_shot,
                    text,
                    prompt_text,
                    str(reference_path),
                    stream=stream,
                    speed=float(speed),
                )
        else:
            generator = self._call_with_supported_kwargs(
                model.inference_sft,
                text,
                speaker,
                stream=stream,
                speed=float(speed),
            )

        for item in generator:
            chunk = item.get("tts_speech") if isinstance(item, dict) else None
            if chunk is not None:
                outputs.append(self._to_np_audio(chunk))

        if not outputs:
            raise RuntimeError("CosyVoice2 generated no audio.")

        audio = np.concatenate(outputs)
        sf.write(output_path, audio, int(self.sample_rate))
        duration_seconds = float(len(audio) / self.sample_rate) if self.sample_rate > 0 else None
        return {
            "output_path": output_path,
            "sample_rate": int(self.sample_rate),
            "duration_seconds": duration_seconds,
            "language": language,
            "stream": stream,
        }
