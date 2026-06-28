import os
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline


class Kandinsky22Runner:
    def __init__(self):
        self.decoder_path = os.getenv("KANDINSKY22_DECODER_MODEL_PATH", "/gpt-lab/long/models/text-to-image/kandinsky-2-2-decoder")
        self.prior_path = os.getenv("KANDINSKY22_PRIOR_MODEL_PATH", "/gpt-lab/long/models/text-to-image/kandinsky-2-2-prior")
        self.prior = None
        self.decoder = None

    def load(self):
        if self.prior is not None and self.decoder is not None:
            return self.prior, self.decoder
        if not os.path.isdir(self.decoder_path):
            raise FileNotFoundError(f"Kandinsky 2.2 decoder folder not found: {self.decoder_path}")
        if not os.path.isdir(self.prior_path):
            raise FileNotFoundError(f"Kandinsky 2.2 prior folder not found: {self.prior_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Kandinsky 2.2 needs GPU inference.")
        self.prior = KandinskyV22PriorPipeline.from_pretrained(self.prior_path, torch_dtype=torch.float16, local_files_only=True).to("cuda")
        self.decoder = KandinskyV22Pipeline.from_pretrained(self.decoder_path, torch_dtype=torch.float16, local_files_only=True).to("cuda")
        return self.prior, self.decoder

    def generate(self, prompt: str, output_path: str, negative_prompt: str | None = None, width: int = 1024, height: int = 1024, steps: int = 50, guidance_scale: float = 4.0, seed: int | None = None, num_images: int = 1) -> str:
        prior, decoder = self.load()
        generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
        prior_output = prior(prompt=prompt, negative_prompt=negative_prompt or "", generator=generator, num_images_per_prompt=int(num_images))
        image = decoder(prompt=prompt, image_embeds=prior_output.image_embeds, negative_image_embeds=prior_output.negative_image_embeds, width=width, height=height, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator).images[0]
        image.save(output_path)
        return output_path
