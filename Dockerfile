# file: Dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/transformers \
    HF_DATASETS_CACHE=/runpod-volume/huggingface-cache/datasets \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y \
    git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
