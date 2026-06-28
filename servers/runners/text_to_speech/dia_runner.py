import os
import numpy as np
import soundfile as sf

DEFAULT_SAMPLE_RATE = 44100


class DiaRunner:
    def __init__(self):
        self.model_path = os.getenv("DIA_MODEL_PATH", "/gpt-lab/long/models/text-to-speech/dia-1-6b")
        self.model = None

    def load(self):
        if self.model is not None:
            return self.model
        import torch
        from dia.model import Dia

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Dia model folder not found: {self.model_path}")
        config_path = os.path.join(self.model_path, "config.json")
        checkpoint_path = os.path.join(self.model_path, "dia-v0_1.pth")
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = os.path.join(self.model_path, "model.safetensors")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compute_dtype = "float16" if torch.cuda.is_available() else "float32"
        self.model = Dia.from_local(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            compute_dtype=compute_dtype,
            device=device,
        )
        return self.model

    def generate(self, text: str, output_path: str, audio_prompt_path: str | None = None):
        model = self.load()
        if "[S1]" not in text and "[S2]" not in text:
            text = f"[S1] {text}"
        output = model.generate(
            text,
            audio_prompt=audio_prompt_path or None,
            use_torch_compile=False,
            verbose=False,
        )
        audio = np.asarray(output, dtype=np.float32).squeeze()
        sf.write(output_path, audio, DEFAULT_SAMPLE_RATE)
        return {"sample_rate": DEFAULT_SAMPLE_RATE}
