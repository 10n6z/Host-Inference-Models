FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY services/audio-csm/requirements.txt /build/requirements.txt
RUN pip install --prefix=/install --no-cache-dir -r /build/requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY servers/ /app/servers/
COPY services/common/ /app/services/common/
COPY services/audio-csm/main.py /app/services/audio-csm/main.py

ENV PYTHONPATH=/app/services:/app/servers
ENV AUDIO_PUBLIC_BASE_URL=http://audio-csm:8110
ENV DEVICE=cpu

WORKDIR /app/services/audio-csm
EXPOSE 8110

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8110"]
