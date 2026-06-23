from __future__ import annotations

import os
import shutil
from pathlib import Path

import soundfile as sf

CHATTERBOX_MTL_SPACE = os.getenv(
    "CHATTERBOX_MTL_SPACE", "ResembleAI/Chatterbox-Multilingual-TTS-V3"
)

SUPPORTED_LANGUAGES = [
    "en", "ar", "da", "de", "el", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
]

TEXT_MAX_LENGTH = 300

DEFAULT_REFERENCE_AUDIO = os.getenv(
    "CHATTERBOX_MTL_DEFAULT_REF",
    "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
)


class ChatterboxMultilingualRunner:
    """Remote runner for ResembleAI/Chatterbox-Multilingual-TTS-V3 via gradio_client."""

    def __init__(self):
        self.model_id = "ResembleAI/Chatterbox-Multilingual-TTS-V3"
        self.space = CHATTERBOX_MTL_SPACE
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.client = None

    def load(self):
        if self.client is not None:
            return

        from gradio_client import Client

        if self.hf_token:
            self.client = Client(self.space, hf_token=self.hf_token)
        else:
            self.client = Client(self.space)

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        language_id: str = "en",
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        seed_num: int = 0,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> dict:
        self.load()

        from gradio_client import handle_file

        if language_id not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        ref_source = audio_prompt_path or DEFAULT_REFERENCE_AUDIO
        audio_prompt = handle_file(ref_source)

        remote_path = self.client.predict(
            text_input=text,
            audio_prompt_path_input=audio_prompt,
            language_id_input=language_id,
            exaggeration_input=float(exaggeration),
            temperature_input=float(temperature),
            seed_num_input=float(seed_num),
            cfgw_input=float(cfg_weight),
            api_name="/generate_tts_audio",
        )

        if not remote_path or not Path(remote_path).is_file():
            raise RuntimeError("Remote Space returned no audio file.")

        shutil.copyfile(remote_path, output_path)

        info = sf.info(output_path)
        sample_rate = int(info.samplerate)
        duration = float(info.frames / info.samplerate) if info.samplerate else None

        return {
            "output_path": output_path,
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "parameters": {
                "language_id": language_id,
                "audio_prompt_path": ref_source,
                "exaggeration": exaggeration,
                "temperature": temperature,
                "seed_num": seed_num,
                "cfg_weight": cfg_weight,
            },
        }
