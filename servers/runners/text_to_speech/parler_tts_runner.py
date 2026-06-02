from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


def _patch_transformers_compat():
    """Patch transformers 5.x to provide APIs parler-tts expects from 4.46.x."""
    import torch
    import transformers.pytorch_utils as pu

    if not hasattr(pu, "isin_mps_friendly"):
        def isin_mps_friendly(elements, test_elements):
            if hasattr(torch, "isin"):
                return torch.isin(elements, test_elements)
            return elements.unsqueeze(-1).eq(test_elements).any(-1)
        pu.isin_mps_friendly = isin_mps_friendly

    import transformers.cache_utils as cu
    if not hasattr(cu, "SlidingWindowCache"):
        cu.SlidingWindowCache = getattr(cu, "DynamicSlidingWindowLayer", cu.DynamicCache)


class ParlerTTSRunner:
    """Runner for ylacombe/parler-tts-mini-jenny-30H."""

    def __init__(self):
        self.model_id = "ylacombe/parler-tts-mini-jenny-30H"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "ylacombe--parler-tts-mini-jenny-30H"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.tokenizer = None
        self.sample_rate = 44100

    def load(self):
        if self.model is not None:
            return

        _patch_transformers_compat()
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        source = str(self.local_path) if self.local_path else self.model_id
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(source)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.sample_rate = self.model.config.sampling_rate

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        description: str = "A female speaker delivers a clear and natural speech.",
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        input_ids = self.tokenizer(description, return_tensors="pt").input_ids
        prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids

        with torch.no_grad():
            generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)

        audio = generation.cpu().numpy().squeeze()
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "description": description,
            },
        }
