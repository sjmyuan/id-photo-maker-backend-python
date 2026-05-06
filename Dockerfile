# ── Stage 1: dependency install ──────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System libraries required by onnxruntime (rembg) and mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Use pip mirror (Aliyun) - swap to default PyPI if building outside China
COPY pip.conf /etc/pip.conf

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# ── Stage 2: pre-load models ──────────────────────────────────────────────────
FROM deps AS model-preload

# Store models inside the image so the container starts without network access
ENV U2NET_HOME=/app/models

COPY download-models.sh ./
RUN chmod +x download-models.sh && \
    PYTHON=python ./download-models.sh

# ── Stage 3: final runtime image ──────────────────────────────────────────────
FROM model-preload AS runtime

COPY src/ ./src/
COPY main.py ./

EXPOSE 3000

# CORS_ORIGIN and BG_REMOVAL_MODEL can be overridden at runtime
ENV PORT=3000 \
    CORS_ORIGIN="" \
    BG_REMOVAL_MODEL="birefnet-portrait" \
    U2NET_HOME=/app/models

CMD ["python", "main.py"]
