FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV UPSCAYL_VERSION=latest \
    UPSCAYL_CLI_URL=https://github.com/upscayl/upscayl/releases/latest/download/upscayl-cli-linux.tar.xz \
    OUTPUT_DIR=/app/output

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tar \
        xz-utils \
        libgtk-3-0 \
        libnss3 \
        libasound2 \
        libatk-bridge2.0-0 \
        libx11-xcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libxkbcommon0 \
        libpango-1.0-0 \
        libgbm1 \
        libatk1.0-0 \
        wget && \
    rm -rf /var/lib/apt/lists/*

# --- Install Node and build Upscayl CLI from source ---
RUN apt-get update && apt-get install -y \
    git nodejs npm libgl1 libglib2.0-0 wget curl && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/upscayl/upscayl-cli.git /opt/upscayl && \
    cd /opt/upscayl && npm install && npm run build && npm install -g .

