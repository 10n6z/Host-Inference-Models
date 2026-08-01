FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY services/vision-ocr-ensemble/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/vision-common/metrics.py /app/vision_common_metrics.py
COPY services/vision-ocr-ensemble/main.py /app/main.py

ENV VISION_OCR_ENSEMBLE_LOG_LEVEL=info

EXPOSE 8123

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123"]
