import os

import torch
from diffusers import AutoModel, SkyReelsV2DiffusionForcingPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from runners.gpu_utils import has_multiple_cuda_devices, maybe_enable_multi_gpu


class SkyReelsV2DF13BRunner:
    """SkyReels V2 Diffusion-Forcing 1.3B 540P via diffusers.

    Lightweight long-form T2V model (~14.7 GB peak at 540P per the model card).
    Uses bfloat16 transformer + fp32 VAE + UniPC scheduler with flow-shift 8.0
    (T2V default). Loaded once, kept resident.
    """

    def __init__(self):
        self.model_path = os.getenv(
            "SKYREELS_V2_DF_13B_MODEL_PATH",
            "/gpt-lab/long/models/text-to-video/skyreels-v2-df-1.3b-540p",
        )
        self.pipe = None
        self._loaded_shift = None

    def load_model(self, shift: float = 8.0):
        if self.pipe is not None and self._loaded_shift == shift:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"SkyReels V2 DF 1.3B model folder not found: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SkyReels V2 DF 1.3B needs GPU inference.")

        vae = AutoModel.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
        )
        self.pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
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
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        self._loaded_shift = shift
        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str | None = None,
        width: int = 960,
        height: int = 544,
        num_frames: int = 97,
        steps: int = 30,
        guidance_scale: float = 6.0,
        shift: float = 8.0,
        seed: int | None = None,
        fps: int = 24,
        base_num_frames: int = 97,
        ar_step: int = 0,
        causal_block_size: int = 1,
        overlap_history: int | None = None,
        addnoise_condition: float = 20.0,
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
            base_num_frames=int(base_num_frames),
            ar_step=int(ar_step),
            causal_block_size=int(causal_block_size),
            overlap_history=overlap_history,
            addnoise_condition=float(addnoise_condition),
            generator=generator,
        )
        frames = result.frames[0]
        export_to_video(frames, output_path, fps=int(fps))
        return output_path
