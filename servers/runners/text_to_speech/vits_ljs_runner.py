import os
import torch
import soundfile as sf
from transformers import AutoTokenizer, VitsModel


class VITSLJSRunner:
    def __init__(self):
        self.model_path = os.getenv("VITS_LJS_MODEL_PATH", "/gpt-lab/long/models/text-to-speech/vits-ljs")
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"VITS LJSpeech model folder not found: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = VitsModel.from_pretrained(self.model_path, local_files_only=True)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        return self.model, self.tokenizer

    def generate(self, text: str, output_path: str):
        model, tokenizer = self.load()
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.no_grad():
            output = model(**inputs).waveform
        waveform = output.squeeze().detach().cpu().float().numpy()
        sample_rate = int(model.config.sampling_rate)
        sf.write(output_path, waveform, sample_rate)
        return {"sample_rate": sample_rate}
