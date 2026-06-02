from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class OrpheusTTSRunner:
    """Runner for OrpheusTTS/Orpheus-3b-0.1-ft using SNAC audio codec."""

    SNAC_TOKEN_OFFSET = 128266
    AUDIO_TOKEN_COUNT = 4096
    EOS_TOKEN_ID = 128258

    def __init__(self, variant: str = "orpheus-3b"):
        self.variant = variant
        if "canopylabs" in variant or "prod" in variant:
            self.model_id = "canopylabs/orpheus-tts-0.1-finetune-prod"
            self.hf_dir = "canopylabs--orpheus-tts-0.1-finetune-prod"
        else:
            self.model_id = "OrpheusTTS/Orpheus-3b-0.1-ft"
            self.hf_dir = "OrpheusTTS--Orpheus-3b-0.1-ft"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / self.hf_dir
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        self.sample_rate = 24000

    def load(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from snac import SNAC

        source = str(self.local_path) if self.local_path and self.local_path.is_dir() else self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32)
        self.model.eval()

        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model.eval()

    def _redistribute_codes(self, raw_codes: list[int]) -> list[list[int]]:
        """Redistribute flat token sequence into 3-layer SNAC code lists."""
        layer1, layer2, layer3 = [], [], []
        for i in range(len(raw_codes) // 7):
            base = i * 7
            if base + 6 >= len(raw_codes):
                break
            layer1.append(raw_codes[base])
            layer2.append(raw_codes[base + 1])
            layer3.append(raw_codes[base + 2])
            layer3.append(raw_codes[base + 3])
            layer2.append(raw_codes[base + 4])
            layer3.append(raw_codes[base + 5])
            layer3.append(raw_codes[base + 6])
        return [layer1, layer2, layer3]

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 1200,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        prompt = f"<custom_token_3>{text}<custom_token_1>"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        token_list = generated_ids.tolist()

        raw_codes = []
        for t in token_list:
            if t == self.EOS_TOKEN_ID:
                break
            adjusted = t - self.SNAC_TOKEN_OFFSET
            if 0 <= adjusted < self.AUDIO_TOKEN_COUNT:
                raw_codes.append(adjusted)

        if len(raw_codes) < 7:
            raise RuntimeError("Orpheus generated too few audio tokens for SNAC decoding.")

        codes = self._redistribute_codes(raw_codes)
        code_tensors = [torch.tensor(layer, dtype=torch.long).unsqueeze(0) for layer in codes]

        with torch.no_grad():
            audio = self.snac_model.decode(code_tensors)

        audio_np = audio.squeeze().cpu().numpy()
        sf.write(output_path, audio_np, self.sample_rate)
        duration = float(len(audio_np) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
            },
        }
