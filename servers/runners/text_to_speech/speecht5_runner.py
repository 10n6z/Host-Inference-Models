from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


class SpeechT5Runner:
    """Runner for microsoft/speecht5_tts."""

    def __init__(self):
        self.model_id = "microsoft/speecht5_tts"
        local_path = Path(os.getenv("HF_MODELS_ROOT", "models/hf")) / "microsoft--speecht5_tts"
        self.local_path = local_path if local_path.is_dir() else None
        self.model = None
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.sample_rate = 16000

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

        import torch
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

        source = str(self.local_path) if self._has_weights() else self.model_id
        self.processor = SpeechT5Processor.from_pretrained(source)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(source)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        try:
            from datasets import load_dataset
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True
            )
            self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        except Exception:
            torch.manual_seed(0)
            self.speaker_embeddings = torch.randn(1, 512)
        self.sample_rate = 16000

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        speaker_index: int = 7306,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        **kwargs,
    ) -> dict:
        self.load()
        import torch

        if speaker_index != 7306 and self.speaker_embeddings is not None:
            try:
                from datasets import load_dataset
                embeddings_dataset = load_dataset(
                    "Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True
                )
                speaker_emb = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)
            except Exception:
                speaker_emb = self.speaker_embeddings
        else:
            speaker_emb = self.speaker_embeddings

        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_emb,
                vocoder=self.vocoder,
                threshold=threshold,
                minlenratio=minlenratio,
                maxlenratio=maxlenratio,
            )

        audio = speech.cpu().numpy()
        sf.write(output_path, audio, self.sample_rate)
        duration = float(len(audio) / self.sample_rate)
        return {
            "output_path": output_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "speaker_index": speaker_index,
                "threshold": threshold,
                "minlenratio": minlenratio,
                "maxlenratio": maxlenratio,
            },
        }
