import os
import torch
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video
from runners.gpu_utils import has_multiple_cuda_devices, maybe_enable_multi_gpu


class ZeroscopeRunner:
    def __init__(self):
        self.model_path = os.getenv("ZEROSCOPE_MODEL_PATH", "/gpt-lab/long/models/text-to-video/zeroscope-v2-576w")
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return self.pipe
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Zeroscope model folder not found: {self.model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Zeroscope needs GPU inference.")
        self.pipe = TextToVideoSDPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16, local_files_only=True, **({"device_map": "balanced"} if has_multiple_cuda_devices() else {}))
        maybe_enable_multi_gpu(self.pipe)
        return self.pipe

    def generate(self, prompt: str, output_path: str, negative_prompt: str | None = None, width: int = 576, height: int = 320, num_frames: int = 24, steps: int = 40, guidance_scale: float = 9.0, seed: int | None = None, fps: int = 8) -> str:
        pipe = self.load_model()
        generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
        result = pipe(prompt=prompt, negative_prompt=negative_prompt, width=int(width), height=int(height), num_frames=int(num_frames), num_inference_steps=int(steps), guidance_scale=float(guidance_scale), generator=generator)
        export_to_video(result.frames[0], output_path, fps=int(fps))
        return output_path
