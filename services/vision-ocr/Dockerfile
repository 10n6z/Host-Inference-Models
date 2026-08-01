FROM paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY services/vision-ocr/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/vision-common/metrics.py /app/vision_common_metrics.py
COPY services/vision-ocr/main.py /app/main.py

ENV PADDLE_PDX_CACHE_HOME=/gpt-lab/long/models/paddle \
    HOME=/gpt-lab/long/models/paddle \
    HF_HOME=/gpt-lab/long/models/hf \
    VISION_OCR_LOG_LEVEL=info

EXPOSE 8120

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8120"]
