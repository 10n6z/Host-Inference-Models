import os
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from safetensors.torch import load_file


class SDXLLightningRunner:
    def __init__(self):
        self.model_path = os.getenv(
            "SDXL_LIGHTNING_MODEL_PATH",
            "/gpt-lab/long/models/text-to-image/sdxl-lightning",
        )
        self.base_model_path = os.getenv(
            "SDXL_BASE_MODEL_PATH",
            "/gpt-lab/long/models/text-to-image/sdxl-base-1.0",
        )
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"SDXL-Lightning model folder not found: {self.model_path}")
        if not os.path.isdir(self.base_model_path):
            raise FileNotFoundError(f"SDXL base model folder not found: {self.base_model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SDXL-Lightning needs GPU inference.")

        ckpt_path = os.path.join(self.model_path, "sdxl_lightning_4step_unet.safetensors")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SDXL-Lightning 4-step UNet checkpoint not found: {ckpt_path}")

        unet = UNet2DConditionModel.from_config(self.base_model_path, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(load_file(ckpt_path, device="cuda"))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        ).to("cuda")
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        self.pipe.vae.enable_tiling()
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
            generator=generator,
            num_images_per_prompt=int(num_images),
        ).images[0]

        image.save(output_path)
        return output_path
