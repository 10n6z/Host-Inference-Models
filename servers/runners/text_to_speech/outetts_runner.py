from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class OuteTTSRunner:
    """Runner for OuteAI/OuteTTS-0.2-500M and OuteAI/OuteTTS-0.3-1B."""

    def __init__(self, variant: str = "0.2-500M"):
        self.variant = variant
        if "0.3" in variant:
            self.model_id = "OuteAI/OuteTTS-0.3-1B"
            self.hf_dir = "OuteAI--OuteTTS-0.3-1B"
        else:
            self.model_id = "OuteAI/OuteTTS-0.2-500M"
            self.hf_dir = "OuteAI--OuteTTS-0.2-500M"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / self.hf_dir
        self.local_path = local_path if local_path.is_dir() else None
        self.interface = None
        self.sample_rate = 24000

    def load(self):
        if self.interface is not None:
            return

        import outetts

        if hasattr(outetts, "InterfaceVersion"):
            version = outetts.InterfaceVersion.V2
            config = outetts.ModelConfig(
                model_path=self.model_id,
                tokenizer_path=self.model_id,
                interface_version=version,
                backend=outetts.Backend.HF,
            )
            self.interface = outetts.Interface(config)
            return

        version_str = "0.3" if "0.3" in self.variant else "0.2"
        config = outetts.HFModelConfig_v1(
            model_path=self.model_id,
            tokenizer_path=self.model_id,
            device="cpu",
            max_seq_length=4096,
        )
        self.interface = outetts.InterfaceHF(version_str, config)

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
        max_length: int = 4096,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
        **kwargs,
    ) -> dict:
        self.load()
        import outetts

        speaker = None
        cloned = False
        if ref_audio_path and os.path.exists(ref_audio_path):
            if ref_text:
                speaker = self.interface.create_speaker(ref_audio_path, ref_text)
            else:
                speaker = self.interface.create_speaker(ref_audio_path)
            cloned = True

        if hasattr(outetts, "GenerationConfig"):
            sampler = outetts.SamplerConfig(
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            gen_kwargs = dict(text=text, sampler_config=sampler, max_length=max_length)
            if speaker is not None:
                gen_kwargs["speaker"] = speaker
            output = self.interface.generate(outetts.GenerationConfig(**gen_kwargs))
        else:
            output = self.interface.generate(
                text=text,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_length=max_length,
                speaker=speaker,
            )
        output.save(output_path)

        info = sf.info(output_path)
        duration = float(info.frames / info.samplerate)
        return {
            "output_path": output_path,
            "sample_rate": info.samplerate,
            "duration_seconds": duration,
            "parameters": {
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "max_length": max_length,
                "cloned": cloned,
            },
        }
