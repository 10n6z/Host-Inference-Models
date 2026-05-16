import os
import torch
from diffusers import FluxPipeline


class FluxSchnellRunner:
    def __init__(self):
        self.model_id = "black-forest-labs/FLUX.1-schnell"
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        self.pipe = FluxPipeline.from_pretrained(
	    os.getenv("FLUX_MODEL_PATH"),
            torch_dtype=torch.bfloat16,
	    local_files_only=True,
        )

        # Safer default for limited VRAM.
        # Later, if your GPU has enough VRAM, we can optimize this.
        self.pipe.enable_model_cpu_offload()

        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        max_sequence_length: int = 256,
        num_images: int = 1,
    ) -> str:
        pipe = self.load()

        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(seed))

        image = pipe(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            max_sequence_length=int(max_sequence_length),
            num_images_per_prompt=int(num_images),
        ).images[0]

        image.save(output_path)
        return output_path
