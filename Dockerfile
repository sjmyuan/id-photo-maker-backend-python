# ── Stage 1: build — install Python deps into an isolated venv ───────────────
FROM python:3.11-slim AS builder

# binutils provides `strip` for removing debug symbols from .so files
RUN apt-get update && apt-get install -y --no-install-recommends \
        binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY pyproject.toml ./
RUN pip install --no-cache-dir --timeout 600 --retries 5 \
        --index-url https://pypi.org/simple/ . \
    # Remove bytecode cache
    && find /venv -name "*.pyc" -delete \
    && find /venv -type d -name "__pycache__" -print0 | xargs -0 rm -rf \
    # Remove test suites bundled inside packages
    && find /venv -type d \( -name "tests" -o -name "test" \) -print0 | xargs -0 rm -rf \
    # Remove type stubs (not needed at runtime)
    && find /venv -name "*.pyi" -delete \
    # Strip debug symbols from compiled extensions (~30-100 MB savings)
    && find /venv -name "*.so" -print0 | xargs -0 strip --strip-unneeded 2>/dev/null || true

# ── Stage 2: runtime — minimal image, no build tools ─────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# System libraries required by onnxruntime (rembg) and mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgles2 \
        libegl1 \
        libgbm1 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the cleaned venv from the builder — no pip, no build artifacts
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

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
