import os
import torch
from diffusers import Flux2KleinPipeline


class Flux2KleinRunner:
    """FLUX.2 [klein] 4B text-to-image runner.

    Requires diffusers >= 0.39 (Flux2KleinPipeline), installed in the
    dedicated `host-models-flux2` conda env.
    """

    def __init__(self):
        self.model_path = os.getenv(
            "FLUX2_KLEIN_MODEL_PATH",
            "/gpt-lab/long/models/text-to-image/flux-2-klein-4b",
        )
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"FLUX.2 klein model folder not found: {self.model_path}")

        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        # Klein fits in ~13GB VRAM; offload keeps headroom on the shared GPU.
        self.pipe.enable_model_cpu_offload()

        return self.pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int | None = None,
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
            num_images_per_prompt=int(num_images),
        ).images[0]

        image.save(output_path)
        return output_path
