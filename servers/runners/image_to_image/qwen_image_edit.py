import os
import torch
from diffusers import QwenImageEditPlusPipeline
from runners.gpu_utils import has_multiple_cuda_devices


class QwenImageEditRunner:
    """Qwen-Image-Edit-2509 image editing runner (QwenImageEditPlusPipeline).

    Takes one or more input images plus an edit instruction and returns the
    edited image. Available in diffusers >= 0.36 (shared `host-models` env).
    """

    def __init__(self):
        self.model_path = os.getenv(
            "QWEN_IMAGE_EDIT_MODEL_PATH",
            "/gpt-lab/long/models/text-to-image/qwen-image-edit-2509",
        )
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Qwen-Image-Edit model folder not found: {self.model_path}")

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **({"device_map": "balanced"} if has_multiple_cuda_devices() else {}),
        )
        self.pipe.set_progress_bar_config(disable=True)

        if not has_multiple_cuda_devices():
            self.pipe.enable_sequential_cpu_offload()

        return self.pipe

    def generate(
        self,
        prompt: str,
        images: list,
        output_path: str,
        negative_prompt: str = " ",
        steps: int = 40,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        num_images: int = 1,
    ) -> str:
        pipe = self.load()

        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(seed))

        with torch.inference_mode():
            image = pipe(
                image=images,
                prompt=prompt,
                negative_prompt=negative_prompt or " ",
                num_inference_steps=int(steps),
                true_cfg_scale=float(true_cfg_scale),
                guidance_scale=float(guidance_scale),
                generator=generator,
                num_images_per_prompt=int(num_images),
            ).images[0]

        image.save(output_path)
        return output_path
