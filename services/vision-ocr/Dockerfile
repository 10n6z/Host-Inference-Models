FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip

WORKDIR /app

COPY services/vision-ocr/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/vision-ocr/main.py /app/main.py

ENV PADDLE_PDX_CACHE_HOME=/gpt-lab/long/models/paddle \
    HF_HOME=/gpt-lab/long/models/hf \
    VISION_OCR_LOG_LEVEL=info

EXPOSE 8120

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8120"]
