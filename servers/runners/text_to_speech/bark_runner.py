from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class BarkRunner:
    """Runner for suno/bark and suno/bark-small models."""

    def __init__(self, variant: str = "small"):
        self.variant = variant
        self.model_id = f"suno/bark-small" if variant == "small" else "suno/bark"
        hf_dir_name = f"suno--bark-small" if variant == "small" else "suno--bark"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / hf_dir_name
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.processor = None
        self.sample_rate = 24000

    def _has_weights(self) -> bool:
        if not self.local_path:
            return False
        for ext in ("*.safetensors", "*.bin", "*.pth"):
            if list(self.local_path.glob(ext)):
                return True
        return False

    def load(self):
        if self.model is not None:
            return

        from transformers import BarkModel, BarkProcessor

        source = str(self.local_path) if self._has_weights() else self.model_id
        self.processor = BarkProcessor.from_pretrained(source)
        self.model = BarkModel.from_pretrained(source)
        self.sample_rate = self.model.generation_config.sample_rate

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        voice_preset: str = "v2/en_speaker_6",
        do_sample: bool = True,
        temperature: float = 1.0,
        semantic_temperature: float = 1.0,
        coarse_temperature: float = 1.0,
        fine_temperature: float = 1.0,
        semantic_max_new_tokens: int = 768,
        coarse_max_new_tokens: int = 1536,
        fine_max_new_tokens: int = 1536,
        **kwargs,
    ) -> dict:
        self.load()
        import copy
        import torch
        from transformers import GenerationConfig

        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")

        semantic_dict = copy.deepcopy(self.model.generation_config.semantic_config)
        semantic_dict["max_new_tokens"] = semantic_max_new_tokens
        semantic_dict["temperature"] = semantic_temperature
        semantic_dict["do_sample"] = do_sample
        semantic_config = GenerationConfig(**semantic_dict)

        coarse_dict = copy.deepcopy(self.model.generation_config.coarse_acoustics_config)
        coarse_dict["max_new_tokens"] = coarse_max_new_tokens
        coarse_dict["temperature"] = coarse_temperature
        coarse_dict["do_sample"] = do_sample
        coarse_config = GenerationConfig(**coarse_dict)

        fine_dict = copy.deepcopy(self.model.generation_config.fine_acoustics_config)
        fine_dict["temperature"] = fine_temperature
        fine_dict["do_sample"] = do_sample
        fine_config = GenerationConfig(**fine_dict)

        with torch.no_grad():
            audio_array = self.model.generate(
                **inputs,
                semantic_generation_config=semantic_config,
                coarse_generation_config=coarse_config,
                fine_generation_config=fine_config,
            )

        audio_array = audio_array.cpu().numpy().squeeze()
        sf.write(output_path, audio_array, self.sample_rate)
        duration = float(len(audio_array) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "voice_preset": voice_preset,
                "do_sample": do_sample,
                "temperature": temperature,
                "semantic_temperature": semantic_temperature,
                "coarse_temperature": coarse_temperature,
                "fine_temperature": fine_temperature,
                "semantic_max_new_tokens": semantic_max_new_tokens,
                "coarse_max_new_tokens": coarse_max_new_tokens,
                "fine_max_new_tokens": fine_max_new_tokens,
            },
        }
