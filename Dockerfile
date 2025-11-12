# Use the stable, pre-built PyTorch image
ARG BASE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

# ---- Environment ----
ENV OUTPUT_DIR=/app/output \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

# ---- Create directories ----
RUN mkdir -p "$OUTPUT_DIR" "$HUGGINGFACE_HUB_CACHE"

WORKDIR /app

# ---- Install Python deps ----
COPY requirements.txt ./

#  NOTE: this base already has torch + torchvision installed.
#  Do NOT reinstall them — just the ESRGAN stack.
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y numpy || true && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade numpy==1.26.4 && \
    sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|' \
        $(python -c "import basicsr, os; print(os.path.dirname(basicsr.__file__))")/data/degradations.py || true

# ---- Copy code ----
COPY . .

# ---- Default command ----
CMD ["python", "server.py"]
