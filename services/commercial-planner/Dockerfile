FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY services/commercial-planner/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/vision-common/vision_common_metrics.py /app/vision_common_metrics.py
COPY services/commercial-planner/main.py /app/main.py
COPY services/commercial-planner/metadata.py /app/metadata.py
COPY services/commercial-planner/vision_plan.py /app/vision_plan.py
COPY services/commercial-planner/providers /app/providers

ENV COMMERCIAL_PLANNER_LOG_LEVEL=info \
    PLANNER_PROVIDERS_ENABLED=self_hosted,rules

EXPOSE 8127

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8127"]
