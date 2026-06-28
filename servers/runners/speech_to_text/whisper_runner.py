import base64
import binascii
import os
import tempfile

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


class WhisperASRRunner:
    def __init__(self, env_var: str, default_path: str):
        self.model_path = os.getenv(env_var, default_path)
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Whisper model folder not found: {self.model_path}")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_path, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        return self.model, self.processor

    def _audio_array(self, audio: str):
        if os.path.exists(audio):
            return librosa.load(audio, sr=16000)[0]
        payload = audio.strip()
        if payload.startswith("data:"):
            _, _, payload = payload.partition(",")
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio: invalid base64 data") from exc
        if not data:
            raise ValueError("audio: decoded audio is empty")
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(data)
            handle.flush()
            return librosa.load(handle.name, sr=16000)[0]

    def transcribe(self, audio: str, language: str | None = None, task: str = "transcribe"):
        model, processor = self.load()
        array = self._audio_array(audio)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        inputs = processor(array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(model.device, dtype=dtype)
        generate_kwargs = {}
        if hasattr(processor, "get_decoder_prompt_ids"):
            generate_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(language=language, task=task)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, **generate_kwargs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"text": text.strip(), "language": language, "task": task}


class WhisperLargeV3Runner(WhisperASRRunner):
    def __init__(self):
        super().__init__("WHISPER_LARGE_V3_MODEL_PATH", "/gpt-lab/long/models/speech-to-text/whisper-large-v3")


class DistilWhisperLargeV3Runner(WhisperASRRunner):
    def __init__(self):
        super().__init__("DISTIL_WHISPER_LARGE_V3_MODEL_PATH", "/gpt-lab/long/models/speech-to-text/distil-whisper-large-v3")
