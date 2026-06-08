from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class CSMRunner:
    """Runner for sesame/csm-1b (transformers-native version)."""

    def __init__(self):
        self.model_id = "sesame/csm-1b"
        self.hf_dir = "sesame--csm-1b"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / self.hf_dir
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.processor = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration

        source = str(self.local_path) if self.local_path else self.model_id
        device = os.getenv("DEVICE", "cpu")

        self.processor = AutoProcessor.from_pretrained(source)
        self.model = CsmForConditionalGeneration.from_pretrained(source, device_map=device)

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        speaker: int = 0,
        max_audio_length_ms: int = 10000,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        # Format text with speaker ID prefix
        conversation = [
            {"role": str(speaker), "content": [{"type": "text", "text": text}]},
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.model.device)

        max_new_tokens = int(max_audio_length_ms / 1000 * 12.5 * 32)

        with torch.no_grad():
            audio_out = self.model.generate(
                **inputs,
                output_audio=True,
                max_new_tokens=max_new_tokens,
            )

        # generate returns a list of audio arrays/tensors
        if isinstance(audio_out, list):
            audio_item = audio_out[0]
        else:
            audio_item = audio_out

        if hasattr(audio_item, "cpu"):
            audio_item = audio_item.cpu().numpy()
        audio = np.squeeze(np.asarray(audio_item)).astype(np.float32)
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {"speaker": speaker, "max_audio_length_ms": max_audio_length_ms},
        }
