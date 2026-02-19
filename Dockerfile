# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-no-torch.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir uv

RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --no-cache \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --no-cache -r requirements-no-torch.txt

COPY . .

ENV WATCH_FOLDER=/data/input
ENV OUTPUT_FOLDER=/data/output
ENV MODELS_FOLDER=/data/models
ENV LOG_FILE=/data/logs/app.log
ENV FORCE_CPU=true
ENV FRAME_FAKE_THRESHOLD=0.5
ENV VIDEO_FAKE_THRESHOLD=0.4

RUN mkdir -p /data/input /data/output /data/logs /data/models

VOLUME ["/data/input", "/data/output", "/data/logs", "/data/models"]

CMD ["python", "app.py"]
