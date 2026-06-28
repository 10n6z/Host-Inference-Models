import os
import torch
from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline


class WuerstchenV2Runner:
    def __init__(self):
        self.decoder_path = os.getenv("WUERSTCHEN_V2_MODEL_PATH", "/gpt-lab/long/models/text-to-image/wuerstchen-v2")
        self.prior_path = os.getenv("WUERSTCHEN_PRIOR_MODEL_PATH", "/gpt-lab/long/models/text-to-image/wuerstchen-prior")
        self.prior = None
        self.decoder = None

    def load(self):
        if self.prior is not None and self.decoder is not None:
            return self.prior, self.decoder
        if not os.path.isdir(self.decoder_path):
            raise FileNotFoundError(f"Würstchen decoder folder not found: {self.decoder_path}")
        if not os.path.isdir(self.prior_path):
            raise FileNotFoundError(f"Würstchen prior folder not found: {self.prior_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Würstchen needs GPU inference.")
        self.prior = WuerstchenPriorPipeline.from_pretrained(self.prior_path, torch_dtype=torch.float16, local_files_only=True).to("cuda")
        self.decoder = WuerstchenDecoderPipeline.from_pretrained(self.decoder_path, torch_dtype=torch.float16, local_files_only=True).to("cuda")
        return self.prior, self.decoder

    def generate(self, prompt: str, output_path: str, negative_prompt: str | None = None, width: int = 1024, height: int = 1024, steps: int = 30, guidance_scale: float = 4.0, seed: int | None = None, num_images: int = 1) -> str:
        prior, decoder = self.load()
        generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
        prior_output = prior(prompt=prompt, negative_prompt=negative_prompt or "", width=width, height=height, guidance_scale=guidance_scale, num_inference_steps=steps, generator=generator, num_images_per_prompt=int(num_images))
        image = decoder(image_embeddings=prior_output.image_embeddings, prompt=prompt, negative_prompt=negative_prompt or "", guidance_scale=0.0, num_inference_steps=max(1, min(12, steps)), generator=generator).images[0]
        image.save(output_path)
        return output_path
