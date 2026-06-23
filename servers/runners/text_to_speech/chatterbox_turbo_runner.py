from __future__ import annotations

import os
import shutil
from pathlib import Path

import soundfile as sf

CHATTERBOX_TURBO_SPACE = os.getenv(
    "CHATTERBOX_TURBO_SPACE", "ResembleAI/chatterbox-turbo-demo"
)

TEXT_MAX_LENGTH = 300

DEFAULT_REFERENCE_AUDIO = os.getenv(
    "CHATTERBOX_TURBO_DEFAULT_REF",
    "https://storage.googleapis.com/chatterbox-demo-samples/turbo/2.wav",
)

# Emotion / sound event tags the turbo model understands inline in `text`.
EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]",
]


class ChatterboxTurboRunner:
    """Remote runner for ResembleAI/chatterbox-turbo-demo via gradio_client.

    Supports inline event tags (e.g. "[chuckle]", "[sigh]") in the text input.
    """

    def __init__(self):
        self.model_id = "ResembleAI/chatterbox-turbo"
        self.space = CHATTERBOX_TURBO_SPACE
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
        audio_prompt_path: str | None = None,
        temperature: float = 0.8,
        seed_num: int = 0,
        min_p: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
        norm_loudness: bool = True,
        **kwargs,
    ) -> dict:
        self.load()

        from gradio_client import handle_file

        ref_source = audio_prompt_path or DEFAULT_REFERENCE_AUDIO
        audio_prompt = handle_file(ref_source)

        try:
            result = self.client.predict(
                text=text,
                audio_prompt_path=audio_prompt,
                temperature=float(temperature),
                seed_num=float(seed_num),
                min_p=float(min_p),
                top_p=float(top_p),
                top_k=float(top_k),
                repetition_penalty=float(repetition_penalty),
                norm_loudness=bool(norm_loudness),
                api_name="/generate",
            )
        except Exception as exc:
            msg = str(exc)
            if "ZeroGPU" in msg or "quota" in msg.lower():
                raise RuntimeError(
                    "Chatterbox Turbo Space ZeroGPU quota exceeded. "
                    "Set HF_TOKEN to a PRO account or retry later."
                ) from exc
            if msg.strip() in ("AssertionError", ""):
                raise RuntimeError(
                    "Chatterbox Turbo generation failed on the remote Space "
                    "(often caused by a too-short reference audio). "
                    "Provide a longer audio_prompt_path."
                ) from exc
            raise

        remote_path = result.get("path") if isinstance(result, dict) else result
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
                "audio_prompt_path": ref_source,
                "temperature": temperature,
                "seed_num": seed_num,
                "min_p": min_p,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "norm_loudness": norm_loudness,
            },
        }
