import os
import torch
from diffusers import StableDiffusion3Pipeline


class SD35MediumRunner:
    def __init__(self):
        self.model_path = os.getenv(
            "SD35_MODEL_PATH",
            "/home/long/local-ai/models/text-to-image/stable-diffusion-3.5-medium",
        )
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"SD3.5 model folder not found: {self.model_path}")

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        if torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        else:
            raise RuntimeError("CUDA is not available. SD3.5 needs GPU inference.")

        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        guidance_scale: float = 4.5,
        seed: int | None = None,
        max_sequence_length: int = 256,
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
        ).images[0]

        image.save(output_path)
        return output_path
