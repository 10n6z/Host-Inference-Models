import os
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from runners.gpu_utils import has_multiple_cuda_devices, maybe_enable_multi_gpu


class WanI2V14BRunner:
    def __init__(self):
        self.model_path = os.getenv("WAN_I2V_14B_MODEL_PATH", "/gpt-lab/long/models/text-to-video/wan21-i2v-14b-480p")
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return self.pipe
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Wan2.1 I2V 14B model folder not found: {self.model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Wan I2V needs GPU inference.")
        self.pipe = WanImageToVideoPipeline.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, local_files_only=True, **({"device_map": "balanced"} if has_multiple_cuda_devices() else {}))
        maybe_enable_multi_gpu(self.pipe)
        return self.pipe

    def generate(self, prompt: str, image, output_path: str, negative_prompt: str | None = None, width: int = 832, height: int = 480, num_frames: int = 81, steps: int = 40, guidance_scale: float = 5.0, seed: int | None = None, fps: int = 16, last_image=None) -> str:
        pipe = self.load_model()
        generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
        kwargs = {}
        if last_image is not None:
            kwargs["last_image"] = last_image
        result = pipe(prompt=prompt, image=image, negative_prompt=negative_prompt, width=int(width), height=int(height), num_frames=int(num_frames), num_inference_steps=int(steps), guidance_scale=float(guidance_scale), generator=generator, **kwargs)
        export_to_video(result.frames[0], output_path, fps=int(fps))
        return output_path


class WanFLF2V14BRunner(WanI2V14BRunner):
    def __init__(self):
        self.model_path = os.getenv("WAN_FLF2V_14B_MODEL_PATH", "/gpt-lab/long/models/text-to-video/wan21-flf2v-14b-720p")
        self.pipe = None
