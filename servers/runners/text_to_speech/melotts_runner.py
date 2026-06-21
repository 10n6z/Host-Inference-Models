from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf

SUPPORTED_LANGUAGES = ("EN", "ES", "FR", "ZH", "JP", "KR")

DEFAULT_SPEAKER_BY_LANGUAGE = {
    "EN": "EN-US",
    "ES": "ES",
    "FR": "FR",
    "ZH": "ZH",
    "JP": "JP",
    "KR": "KR",
}


class MeloTTSRunner:
    """Multi-language runner for myshell-ai/MeloTTS.

    Mirrors the mrfakename/MeloTTS Gradio Space: one model per language
    (EN/ES/FR/ZH/JP/KR), each exposing named speakers via spk2id.
    """

    def __init__(self):
        self.model_id = "myshell-ai/MeloTTS"
        self._models: dict[str, object] = {}
        self.sample_rate = 44100

    def _load(self, language: str):
        language = self._normalize_language(language)
        model = self._models.get(language)
        if model is not None:
            return model

        from melo.api import TTS

        model = TTS(language=language, device=os.getenv("MELOTTS_DEVICE", "cpu"))
        self._models[language] = model
        self.sample_rate = model.hps.data.sampling_rate
        return model

    @staticmethod
    def _normalize_language(language: str) -> str:
        lang = (language or "EN").strip().upper()
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}."
            )
        return lang

    def list_speakers(self, language: str) -> list[str]:
        """Return the named speakers available for a language (Gradio /load_speakers)."""
        model = self._load(language)
        return list(model.hps.data.spk2id.keys())

    def _resolve_speaker_id(self, model, speaker) -> int:
        spk2id = model.hps.data.spk2id

        if speaker is None or speaker == "":
            return next(iter(spk2id.values()))

        if isinstance(speaker, str) and speaker in spk2id:
            return spk2id[speaker]

        if isinstance(speaker, int) or (isinstance(speaker, str) and speaker.isdigit()):
            idx = int(speaker)
            all_ids = list(spk2id.values())
            return all_ids[min(idx, len(all_ids) - 1)]

        raise ValueError(
            f"Unknown speaker '{speaker}'. Available: {', '.join(spk2id.keys())}."
        )

    def generate(
        self,
        *,
        text: str,
        output_path: str,
        language: str = "EN",
        speaker=None,
        speaker_id=None,
        speed: float = 1.0,
        **kwargs,
    ) -> dict:
        language = self._normalize_language(language)
        model = self._load(language)

        selected = speaker if speaker not in (None, "") else speaker_id
        if selected in (None, ""):
            selected = DEFAULT_SPEAKER_BY_LANGUAGE[language]
        resolved_speaker_id = self._resolve_speaker_id(model, selected)

        model.tts_to_file(text, resolved_speaker_id, output_path, speed=speed)

        info = sf.info(output_path)
        duration = float(info.frames / info.samplerate)
        return {
            "output_path": output_path,
            "sample_rate": info.samplerate,
            "duration_seconds": duration,
            "parameters": {
                "language": language,
                "speaker": selected,
                "speed": speed,
            },
        }
