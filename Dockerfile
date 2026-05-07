# ── Stage 1: dependency install ──────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System libraries required by onnxruntime (rembg) and mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgles2 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Use pip mirror (Aliyun) - swap to default PyPI if building outside China
COPY pip.conf /etc/pip.conf

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# ── Stage 2: final runtime image ──────────────────────────────────────────────
FROM deps AS runtime

COPY download-models.sh entrypoint.sh ./
RUN chmod +x download-models.sh entrypoint.sh

COPY src/ ./src/
COPY main.py ./

# Models are stored in a mounted volume, not baked into the image.
# Mount a host directory or named volume at /app/models.
# The entrypoint downloads missing models into the volume on first start.
VOLUME /app/models

EXPOSE 3000

# CORS_ORIGIN and BG_REMOVAL_MODEL can be overridden at runtime
ENV PORT=3000 \
    CORS_ORIGIN="" \
    BG_REMOVAL_MODEL="birefnet-portrait" \
    U2NET_HOME=/app/models

ENTRYPOINT ["./entrypoint.sh"]
