FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY services/multimodal-legacy/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY servers/ /app/servers/

ENV PYTHONPATH=/app/servers
ENV PUBLIC_BASE_URL=http://multimodal-legacy:8001

WORKDIR /app/servers
EXPOSE 8001

CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8001"]
