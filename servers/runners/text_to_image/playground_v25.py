import os
import torch
from diffusers import StableDiffusionXLPipeline


class PlaygroundV25Runner:
    def __init__(self):
        self.model_path = os.getenv("PLAYGROUND_V25_MODEL_PATH", "/gpt-lab/long/models/text-to-image/playground-v2-5")
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Playground v2.5 model folder not found: {self.model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Playground v2.5 needs GPU inference.")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True).to("cuda")
        self.pipe.vae.enable_tiling()
        return self.pipe

    def generate(self, prompt: str, output_path: str, negative_prompt: str | None = None, width: int = 1024, height: int = 1024, steps: int = 30, guidance_scale: float = 3.0, seed: int | None = None, num_images: int = 1) -> str:
        pipe = self.load()
        generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator, num_images_per_prompt=int(num_images)).images[0]
        image.save(output_path)
        return output_path
