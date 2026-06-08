FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY services/audio-espnet/requirements.txt /build/requirements.txt
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

RUN python -m nltk.downloader -d /usr/local/share/nltk_data \
    averaged_perceptron_tagger \
    averaged_perceptron_tagger_eng \
    cmudict

COPY servers/ /app/servers/
COPY services/common/ /app/services/common/
COPY services/audio-espnet/main.py /app/services/audio-espnet/main.py

ENV PYTHONPATH=/app/services:/app/servers
ENV AUDIO_PUBLIC_BASE_URL=http://audio-espnet:8103
ENV DEVICE=cpu

WORKDIR /app/services/audio-espnet
EXPOSE 8103

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8103"]
