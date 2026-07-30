import os
import torch
from diffusers import StableDiffusionInpaintPipeline


class SD15InpaintRunner:
    """Stable Diffusion 1.5 inpainting/outpainting runner.

    Takes a base image plus a binary mask (white = regenerate, black = keep)
    and a prompt, returning the inpainted image. Outpainting is the same call
    with the new canvas padded and its border region marked white in the mask.
    Uses StableDiffusionInpaintPipeline (diffusers, shared `host-models` env).
    """

    def __init__(self):
        self.model_path = os.getenv(
            "SD15_INPAINT_MODEL_PATH",
            "/gpt-lab/long/models/text-to-image/sd-1-5-inpainting",
        )
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"SD1.5 inpaint model folder not found: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SD1.5 inpaint needs GPU inference.")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True,
        )
        self.pipe.to("cuda:0")
        self.pipe.set_progress_bar_config(disable=True)
        return self.pipe

    def generate(
        self,
        prompt: str,
        image,
        mask_image,
        output_path: str,
        negative_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 1.0,
        seed: int | None = None,
        num_images: int = 1,
    ) -> str:
        pipe = self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(int(seed))

        base = image.resize((width, height))
        mask = mask_image.resize((width, height)).convert("L")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=base,
                mask_image=mask,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                strength=float(strength),
                generator=generator,
                num_images_per_prompt=int(num_images),
            ).images[0]

        result.save(output_path)
        return output_path
