import os
import torch
from diffusers import StableDiffusionPipeline
from runners.gpu_utils import has_multiple_cuda_devices, maybe_enable_multi_gpu


class SD21Runner:
    def __init__(self):
        self.model_path = os.getenv(
            "SD21_MODEL_PATH",
            "/gpt-lab/long/models/text-to-image/sd-2-1",
        )
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"SD2.1 model folder not found: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SD2.1 needs GPU inference.")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
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
        height: int = 768,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        num_images: int = 1,
    ) -> str:
        pipe = self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(int(seed))

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=int(num_images),
        ).images[0]

        image.save(output_path)
        return output_path
