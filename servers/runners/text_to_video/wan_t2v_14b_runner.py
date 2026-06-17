import os

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video
from runners.gpu_utils import has_multiple_cuda_devices, maybe_enable_multi_gpu


class WanT2V14BRunner:
    """Wan2.1 T2V 14B via diffusers WanPipeline.

    Larger sibling of the 1.3B runner. Uses fp32 VAE + bf16 transformer (the
    combination the Wan authors validate), a UniPC scheduler whose flow-shift is
    tuned per-resolution, and model-cpu-offload + VAE tiling so the 14B weights
    fit on a single L40S (~13GB active) without holding the whole model resident.
    """

    def __init__(self):
        self.model_path = os.getenv(
            "WAN_T2V_14B_MODEL_PATH",
            "/gpt-lab/long/models/text-to-video/Wan2.1-T2V-14B-Diffusers",
        )
        self.pipe = None
        self._loaded_shift = None

    def load_model(self, shift: float = 5.0):
        # flow-shift is baked into the scheduler at load time, so rebuild the
        # pipeline only when the requested shift actually changes.
        if self.pipe is not None and self._loaded_shift == shift:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Wan2.1 T2V 14B model folder not found: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Wan2.1 T2V 14B needs GPU inference.")

        vae = AutoencoderKLWan.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
        )
        self.pipe = WanPipeline.from_pretrained(
            self.model_path,
            vae=vae,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **({"device_map": "balanced"} if has_multiple_cuda_devices() else {}),
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=shift
        )
        maybe_enable_multi_gpu(self.pipe)
        self.pipe.vae.enable_tiling()
        self._loaded_shift = shift
        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str | None = None,
        width: int = 832,
        height: int = 480,
        num_frames: int = 81,
        steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 5.0,
        seed: int | None = None,
        fps: int = 16,
    ) -> str:
        pipe = self.load_model(shift=shift)

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
