import os
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from runners.gpu_utils import has_multiple_cuda_devices, maybe_enable_multi_gpu


class HunyuanVideoRunner:
    def __init__(self):
        self.model_path = os.getenv("HUNYUANVIDEO_MODEL_PATH", "/gpt-lab/long/models/text-to-video/hunyuanvideo")
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return self.pipe
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"HunyuanVideo model folder not found: {self.model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. HunyuanVideo needs GPU inference.")
        self.pipe = HunyuanVideoPipeline.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, local_files_only=True, **({"device_map": "balanced"} if has_multiple_cuda_devices() else {}))
        maybe_enable_multi_gpu(self.pipe)
        return self.pipe

    def generate(self, prompt: str, output_path: str, negative_prompt: str | None = None, width: int = 720, height: int = 1280, num_frames: int = 129, steps: int = 30, guidance_scale: float = 6.0, seed: int | None = None, fps: int = 24) -> str:
        pipe = self.load_model()
        generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
        result = pipe(prompt=prompt, negative_prompt=negative_prompt, width=int(width), height=int(height), num_frames=int(num_frames), num_inference_steps=int(steps), guidance_scale=float(guidance_scale), generator=generator)
        export_to_video(result.frames[0], output_path, fps=int(fps))
        return output_path
