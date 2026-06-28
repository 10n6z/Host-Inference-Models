import os
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_video
from runners.gpu_utils import has_multiple_cuda_devices, maybe_enable_multi_gpu


class AnimateDiffV3Runner:
    def __init__(self):
        self.adapter_path = os.getenv("ANIMATEDIFF_V3_MODEL_PATH", "/gpt-lab/long/models/text-to-video/animatediff-v3")
        self.base_model_path = os.getenv("SD15_MODEL_PATH", "/gpt-lab/long/models/text-to-image/sd-v1-5")
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return self.pipe
        if not os.path.isdir(self.adapter_path):
            raise FileNotFoundError(f"AnimateDiff adapter folder not found: {self.adapter_path}")
        if not os.path.isdir(self.base_model_path):
            raise FileNotFoundError(f"SD1.5 base model folder not found: {self.base_model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. AnimateDiff needs GPU inference.")
        adapter = MotionAdapter.from_pretrained(self.adapter_path, torch_dtype=torch.float16, local_files_only=True)
        self.pipe = AnimateDiffPipeline.from_pretrained(self.base_model_path, motion_adapter=adapter, torch_dtype=torch.float16, local_files_only=True, **({"device_map": "balanced"} if has_multiple_cuda_devices() else {}))
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, beta_schedule="linear", clip_sample=False, timestep_spacing="linspace", steps_offset=1)
        maybe_enable_multi_gpu(self.pipe)
        return self.pipe

    def generate(self, prompt: str, output_path: str, negative_prompt: str | None = None, width: int = 512, height: int = 512, num_frames: int = 16, steps: int = 25, guidance_scale: float = 7.5, seed: int | None = None, fps: int = 8) -> str:
        pipe = self.load_model()
        generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
        result = pipe(prompt=prompt, negative_prompt=negative_prompt, width=int(width), height=int(height), num_frames=int(num_frames), num_inference_steps=int(steps), guidance_scale=float(guidance_scale), generator=generator)
        export_to_video(result.frames[0], output_path, fps=int(fps))
        return output_path
