# Use stable PyTorch 2.1.2 base
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# Install Python packages with specific versions for compatibility
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
