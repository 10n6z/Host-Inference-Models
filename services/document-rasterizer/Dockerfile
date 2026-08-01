FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libtiff for Pillow's TIFF plugin.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff6 \
    ca-certificates \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY services/document-rasterizer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/vision-common/vision_common_metrics.py /app/vision_common_metrics.py
COPY services/document-rasterizer/main.py /app/main.py
COPY services/document-rasterizer/rasterizer.py /app/rasterizer.py

ENV DOCUMENT_RASTERIZER_LOG_LEVEL=info \
    GATEWAY_ASSETS_ROOT=/gpt-lab/long/computer-vision/assets

EXPOSE 8130

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8130"]
