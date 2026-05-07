#!/usr/bin/env bash
# Entrypoint: download AI models into the mounted volume if not already present,
# then start the server.
set -euo pipefail

MODEL="${BG_REMOVAL_MODEL:-birefnet-portrait}"
U2NET_DIR="${U2NET_HOME:-/app/models}"

mkdir -p "$U2NET_DIR"

# Check if the model ONNX file already exists in the volume
model_files=$(find "$U2NET_DIR" -name "${MODEL}*.onnx" 2>/dev/null || true)

if [ -z "$model_files" ]; then
    echo "Model '$MODEL' not found in $U2NET_DIR — downloading now..."
    PYTHON=python ./download-models.sh
else
    echo "Model '$MODEL' found in $U2NET_DIR — skipping download."
fi

exec python main.py
