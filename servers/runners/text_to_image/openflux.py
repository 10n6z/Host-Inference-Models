import os
import torch
from diffusers import FluxPipeline


class OpenFluxRunner:
    def __init__(self):
        self.model_path = os.getenv(
            "OPENFLUX_MODEL_PATH",
            "/home/long/local-ai/models/text-to-image/openflux-1",
        )
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"OpenFLUX model folder not found: {self.model_path}")

        self.pipe = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        if torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        else:
            raise RuntimeError("CUDA is not available. OpenFLUX needs GPU inference.")

        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int | None = None,
        max_sequence_length: int = 256,
        num_images: int = 1,
    ) -> str:
        pipe = self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(int(seed))

        image = pipe(
            prompt=prompt,
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
