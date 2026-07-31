FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# eng/fin/swe language data, pinned per plan Task 5. CPU-only: no CUDA base
# image, no GPU device reservation in docker-compose.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fin \
    tesseract-ocr-swe \
    ca-certificates \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY services/vision-tesseract/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/vision-tesseract/main.py /app/main.py

ENV VISION_TESSERACT_LOG_LEVEL=info

EXPOSE 8122

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8122"]
