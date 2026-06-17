import os

import torch
from diffusers import LTXPipeline
from runners.gpu_utils import has_multiple_cuda_devices, maybe_enable_multi_gpu
from diffusers.utils import export_to_video


class LTXVideoRunner:
    def __init__(self):
        self.model_path = os.getenv(
            "LTX_VIDEO_MODEL_PATH",
            "/home/long/local-ai/models/text-to-video/LTX-Video",
        )
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"LTX-Video model folder not found: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. LTX-Video needs GPU inference.")

        self.pipe = LTXPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **({"device_map": "balanced"} if has_multiple_cuda_devices() else {}),
        )
        maybe_enable_multi_gpu(self.pipe)

        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str | None = None,
        width: int = 768,
        height: int = 432,
        num_frames: int = 97,
        steps: int = 30,
        guidance_scale: float = 3.0,
        seed: int | None = None,
        fps: int = 24,
        decode_timestep: float = 0.03,
        decode_noise_scale: float = 0.025,
    ) -> str:
        pipe = self.load_model()

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            decode_timestep=float(decode_timestep),
            decode_noise_scale=float(decode_noise_scale),
        )
        frames = result.frames[0]
        export_to_video(frames, output_path, fps=int(fps))
        return output_path
