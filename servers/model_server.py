import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from runners.text_to_image.flux_schnell import FluxSchnellRunner
from runners.text_to_speech.kokoro_runner import KokoroRunner


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

HF_HOME = Path(
    os.getenv("HF_HOME", BASE_DIR.parent / "models" / "hf-cache")
).resolve()
HF_HUB_CACHE = Path(
    os.getenv("HF_HUB_CACHE", HF_HOME / "hub")
).resolve()

OUTPUT_ROOT = Path(
    os.getenv("OUTPUT_ROOT", BASE_DIR.parent / "outputs")
).resolve()

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8001").rstrip("/")

IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "images"
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audio"

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

app = FastAPI(title="Local AI GPU Inference Server")

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")


flux_runner = FluxSchnellRunner()
kokoro_runner = KokoroRunner()


class FluxRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    width: int = Field(1024, ge=512, le=1024)
    height: int = Field(1024, ge=512, le=1024)
    steps: int = Field(4, ge=1, le=8)
    seed: Optional[int] = None


class KokoroRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    lang_code: str = "a"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "local-ai-gpu-inference-server",
    }


@app.get("/models")
def models():
    return [
        {
            "id": "flux-1-schnell",
            "name": "FLUX.1-schnell",
            "modality": "text-to-image",
            "endpoint": "/generate/image/flux",
            "fields": [
                {"name": "prompt", "type": "textarea", "required": True},
                {"name": "width", "type": "number", "default": 1024, "min": 512, "max": 1024},
                {"name": "height", "type": "number", "default": 1024, "min": 512, "max": 1024},
                {"name": "steps", "type": "number", "default": 4, "min": 1, "max": 8},
                {"name": "seed", "type": "number", "required": False},
            ],
        },
        {
            "id": "kokoro-82m",
            "name": "Kokoro-82M",
            "modality": "text-to-speech",
            "endpoint": "/generate/tts/kokoro",
            "fields": [
                {"name": "text", "type": "textarea", "required": True},
                {"name": "voice", "type": "select", "default": "af_heart"},
                {"name": "speed", "type": "number", "default": 1.0, "min": 0.5, "max": 2.0},
                {"name": "lang_code", "type": "select", "default": "a", "options": ["a", "b"]},
            ],
        },
    ]


@app.post("/generate/image/flux")
def generate_flux(req: FluxRequest):
    output_id = f"img_{uuid.uuid4().hex}.png"
    output_path = IMAGE_OUTPUT_DIR / output_id

    try:
        flux_runner.generate(
            prompt=req.prompt,
            output_path=str(output_path),
            width=req.width,
            height=req.height,
            steps=req.steps,
            seed=req.seed,
        )

        output_url = f"{PUBLIC_BASE_URL}/outputs/images/{output_id}"

        return {
            "status": "completed",
            "modelId": "flux-1-schnell",
            "modality": "text-to-image",
            "outputType": "image",
            "outputId": output_id,
            "outputUrl": output_url,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/tts/kokoro")
def generate_kokoro(req: KokoroRequest):
    output_id = f"aud_{uuid.uuid4().hex}.wav"
    output_path = AUDIO_OUTPUT_DIR / output_id

    try:
        kokoro_runner.generate(
            text=req.text,
            output_path=str(output_path),
            voice=req.voice,
            speed=req.speed,
            lang_code=req.lang_code,
        )

        output_url = f"{PUBLIC_BASE_URL}/outputs/audio/{output_id}"

        return {
            "status": "completed",
            "modelId": "kokoro-82m",
            "modality": "text-to-speech",
            "outputType": "audio",
            "outputId": output_id,
            "outputUrl": output_url,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
