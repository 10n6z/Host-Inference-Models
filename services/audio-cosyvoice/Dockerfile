FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY services/audio-cosyvoice/requirements.txt /build/requirements.txt
RUN pip install --prefix=/install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r /build/requirements.txt

RUN pip install --prefix=/install --no-cache-dir "setuptools<81" \
    && PYTHONPATH=/install/lib/python3.10/site-packages \
    pip install --prefix=/install --no-cache-dir --no-build-isolation \
    openai-whisper==20240930

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    sox \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice \
    && rm -rf /app/CosyVoice/.git /app/CosyVoice/third_party/Matcha-TTS/.git

COPY servers/ /app/servers/
COPY services/common/ /app/services/common/
COPY services/audio-cosyvoice/main.py /app/services/audio-cosyvoice/main.py
COPY services/audio-cosyvoice/assets/ /app/services/audio-cosyvoice/assets/

ENV PYTHONPATH=/app/services:/app/servers:/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS
ENV AUDIO_PUBLIC_BASE_URL=http://audio-cosyvoice:8113
ENV DEVICE=cuda

WORKDIR /app/services/audio-cosyvoice
EXPOSE 8113

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8113"]
