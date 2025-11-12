ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

ENV OUTPUT_DIR=/app/output \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

RUN mkdir -p "$OUTPUT_DIR" "$HUGGINGFACE_HUB_CACHE"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade numpy

COPY . .

CMD ["python", "server.py"]
