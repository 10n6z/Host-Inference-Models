import os
import torch
from diffusers import StableDiffusion3Pipeline


class SD35LargeTurboRunner:
    def __init__(self):
        self.model_path = os.getenv(
            "SD35_LARGE_TURBO_MODEL_PATH",
            "/gpt-lab/long/models/text-to-image/sd-3.5-large-turbo",
        )
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"SD3.5 Large Turbo model folder not found: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SD3.5 Large Turbo needs GPU inference.")

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        max_sequence_length: int = 512,
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
            max_sequence_length=max_sequence_length,
            generator=generator,
            num_images_per_prompt=int(num_images),
        ).images[0]

        image.save(output_path)
        return output_path
