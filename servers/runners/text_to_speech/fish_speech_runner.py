from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class FishSpeechRunner:
    def __init__(self):
        self.model_path = Path(
            os.getenv(
                "FISH_SPEECH_MODEL_PATH",
                "/home/long/local-ai/models/text-to-speech/fish-speech-v1.5",
            )
        )
        self.decoder_checkpoint_path = Path(
            os.getenv(
                "FISH_SPEECH_DECODER_CHECKPOINT_PATH",
                str(self.model_path / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
            )
        )
        self.decoder_config_name = os.getenv("FISH_SPEECH_DECODER_CONFIG_NAME", "modded_dac_vq")
        self.device_preference = os.getenv("FISH_SPEECH_DEVICE", "cuda")
        self.use_half = os.getenv("FISH_SPEECH_HALF", "false").lower() == "true"
        self.compile = os.getenv("FISH_SPEECH_COMPILE", "false").lower() == "true"

        self.engine = None
        self.sample_rate = 44100

    def _resolve_decoder_checkpoint(self) -> Path:
        if self.decoder_checkpoint_path.exists():
            return self.decoder_checkpoint_path

        fallback = self.model_path / "codec.pth"
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            "Fish Speech decoder checkpoint not found. "
            f"Tried: {self.decoder_checkpoint_path} and {fallback}"
        )

    def load(self):
        if self.engine is not None:
            return self.engine

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Fish Speech model folder not found: {self.model_path}")

        decoder_checkpoint = self._resolve_decoder_checkpoint()

        try:
            import torch
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        except Exception as exc:
            raise FileNotFoundError(
                "Fish Speech runtime dependencies not available. "
                "Install fish-speech package/repo runtime before using this runner."
            ) from exc

        precision = torch.half if self.use_half else torch.bfloat16
        device = self.device_preference
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(self.model_path),
            device=device,
            precision=precision,
            compile=self.compile,
        )
        decoder_model = load_decoder_model(
            config_name=self.decoder_config_name,
            checkpoint_path=str(decoder_checkpoint),
            device=device,
        )

        if hasattr(decoder_model, "spec_transform"):
            self.sample_rate = int(decoder_model.spec_transform.sample_rate)
        elif hasattr(decoder_model, "sample_rate"):
            self.sample_rate = int(decoder_model.sample_rate)

        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=self.compile,
        )
        return self.engine

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        language: str = "en",
        voice: str = "default",
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int | None = None,
        reference_audio_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if format.lower() != "wav":
            raise ValueError("Fish Speech runner currently supports format=wav only.")

        engine = self.load()

        # Fish Speech local engine decides voice by reference_id/references.
        # Keep current API contract: map reference_audio_id if provided.
        reference_id = reference_audio_id if reference_audio_id else None
        if reference_id is None and voice and voice != "default":
            reference_id = voice

        try:
            from fish_speech.utils.schema import ServeTTSRequest
        except Exception as exc:
            raise FileNotFoundError("fish_speech.utils.schema import failed.") from exc

        req = ServeTTSRequest(
            text=text,
            references=[],
            reference_id=reference_id,
            format="wav",
            latency="normal",
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
            streaming=False,
            use_memory_cache="off",
            seed=None,
            normalize=True,
        )

        audio_np = None
        sample_rate_used = int(sample_rate) if sample_rate is not None else int(self.sample_rate)

        for result in engine.inference(req):
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "final" and isinstance(result.audio, tuple):
                sr, audio = result.audio
                sample_rate_used = int(sr)
                audio_np = audio
                break

        if audio_np is None:
            raise RuntimeError("Fish Speech generated no audio.")

        if not isinstance(audio_np, np.ndarray):
            audio_np = np.array(audio_np)

        sf.write(output_path, audio_np, sample_rate_used)
        duration_seconds = float(len(audio_np) / sample_rate_used) if sample_rate_used > 0 else None
        return {
            "output_path": output_path,
            "sample_rate": sample_rate_used,
            "duration_seconds": duration_seconds,
            "language": language,
            "speed": speed,
        }
