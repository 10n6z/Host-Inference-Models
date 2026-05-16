import numpy as np
import soundfile as sf
from kokoro import KPipeline


class KokoroRunner:
    def __init__(self):
        self.pipelines = {}

    def load(self, lang_code: str = "a"):
        if lang_code not in self.pipelines:
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code)

        return self.pipelines[lang_code]

    def generate(
        self,
        text: str,
        output_path: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
        sample_rate: int = 24000,
    ) -> dict:
        pipeline = self.load(lang_code=lang_code)

        generator = pipeline(
            text,
            voice=voice,
            speed=float(speed),
            split_pattern=r"\n+",
        )

        audio_segments = []

        for _, _, audio in generator:
            audio_segments.append(audio)

        if not audio_segments:
            raise RuntimeError("Kokoro generated no audio.")

        audio = np.concatenate(audio_segments)
        sf.write(output_path, audio, int(sample_rate))

        duration_seconds = float(len(audio) / sample_rate) if sample_rate > 0 else None
        return {
            "output_path": output_path,
            "sample_rate": int(sample_rate),
            "duration_seconds": duration_seconds,
        }
