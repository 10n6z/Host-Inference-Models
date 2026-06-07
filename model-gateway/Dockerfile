FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY model-gateway/requirements.txt /app/model-gateway/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/model-gateway/requirements.txt

COPY model-gateway/ /app/model-gateway/
COPY services/common/ /app/services/common/

ENV PYTHONPATH=/app/services
ENV MODEL_REGISTRY_PATH=/app/model-gateway/registry.yaml

WORKDIR /app/model-gateway
EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
