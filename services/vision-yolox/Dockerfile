FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip

WORKDIR /app

COPY services/vision-yolox/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# yolox's own setup.py declares onnx==1.8.1/onnxruntime==1.8.0/pycocotools,
# which are export/eval-only and unusably old on modern Python -- this
# service only imports yolox.exp/yolox.models/yolox.utils for inference, so
# --no-deps skips them. --no-build-isolation lets its setup.py see the torch
# already installed above (it checks for torch to decide whether to
# pre-compile optional training-only ops, which we don't need either).
RUN pip install --no-cache-dir --no-deps --no-build-isolation "yolox==0.3.0"

COPY services/vision-yolox/main.py /app/main.py

ENV HF_HOME=/gpt-lab/long/models/hf \
    TORCH_HOME=/gpt-lab/long/models/torch \
    VISION_YOLOX_LOG_LEVEL=info \
    VISION_YOLOX_MODEL_NAME=yolox-s

EXPOSE 8125

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8125"]
