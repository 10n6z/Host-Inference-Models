FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY services/audio-melotts/requirements.txt /build/requirements.txt
RUN pip install --prefix=/install --no-cache-dir -r /build/requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    espeak-ng \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY servers/ /app/servers/
COPY services/common/ /app/services/common/
COPY services/audio-melotts/main.py /app/services/audio-melotts/main.py

ENV PYTHONPATH=/app/services:/app/servers
ENV AUDIO_PUBLIC_BASE_URL=http://audio-melotts:8104

WORKDIR /app/services/audio-melotts
EXPOSE 8104

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8104"]
