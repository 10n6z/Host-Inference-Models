import os
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


class ParlerTTSRunner:
    def __init__(self, env_var: str, default_path: str):
        self.model_path = os.getenv(env_var, default_path)
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None

    def load(self):
        if self.model is not None:
            return self.model, self.tokenizer, self.description_tokenizer
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Parler-TTS model folder not found: {self.model_path}")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=dtype, local_files_only=True)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        return self.model, self.tokenizer, self.description_tokenizer

    def generate(self, text: str, output_path: str, description: str | None = None):
        model, tokenizer, description_tokenizer = self.load()
        if not description:
            description = "A clear, natural English speaker with good audio quality."
        prompt_inputs = tokenizer(text, return_tensors="pt")
        description_inputs = description_tokenizer(description, return_tensors="pt")
        if torch.cuda.is_available():
            prompt_inputs = {key: value.to("cuda") for key, value in prompt_inputs.items()}
            description_inputs = {key: value.to("cuda") for key, value in description_inputs.items()}
        with torch.no_grad():
            generation = model.generate(input_ids=description_inputs.input_ids, prompt_input_ids=prompt_inputs.input_ids)
        audio = generation.cpu().numpy().squeeze()
        sample_rate = int(model.config.sampling_rate)
        sf.write(output_path, audio, sample_rate)
        return {"sample_rate": sample_rate}


class ParlerMiniRunner(ParlerTTSRunner):
    def __init__(self):
        super().__init__("PARLER_MINI_MODEL_PATH", "/gpt-lab/long/models/text-to-speech/parler-tts-mini-v1")


class ParlerLargeRunner(ParlerTTSRunner):
    def __init__(self):
        super().__init__("PARLER_LARGE_MODEL_PATH", "/gpt-lab/long/models/text-to-speech/parler-tts-large-v1")
