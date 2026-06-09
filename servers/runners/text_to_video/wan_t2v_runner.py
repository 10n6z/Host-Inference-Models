import os

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video


class WanT2VRunner:
    def __init__(self):
        self.model_path = os.getenv(
            "WAN_T2V_MODEL_PATH",
            "/home/long/local-ai/models/text-to-video/Wan2.1-T2V-1.3B",
        )
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Wan2.1 T2V model folder not found: {self.model_path}")

        self.pipe = WanPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        if torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        else:
            raise RuntimeError("CUDA is not available. Wan2.1 T2V needs GPU inference.")

        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str | None = None,
        width: int = 768,
        height: int = 432,
        num_frames: int = 49,
        steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int | None = None,
        fps: int = 16,
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
        )
        frames = result.frames[0]
        export_to_video(frames, output_path, fps=int(fps))
        return output_path
