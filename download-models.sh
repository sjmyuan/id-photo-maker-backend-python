#!/usr/bin/env bash
# Downloads AI model files required by the backend.
# rembg caches models in ~/.u2net/ (overridable via U2NET_HOME).
# MediaPipe face detector (blaze_face_short_range.tflite) is downloaded separately.
# Run this once before starting the server to avoid download on first request.
#
# The model used is controlled by the BG_REMOVAL_MODEL env var (default: birefnet-portrait).
# Available models: birefnet-portrait, birefnet-general-lite, birefnet-general, u2net, u2net_human_seg, isnet-general-use

set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${BG_REMOVAL_MODEL:-birefnet-portrait}"

# Resolve the cache directory rembg will use
U2NET_DIR="${U2NET_HOME:-$HOME/.u2net}"

# ── Background removal model ─────────────────────────────────────────────────
echo "Checking model: $MODEL"
"$PYTHON" - <<EOF
import os, sys
from pathlib import Path

model = "$MODEL"
u2net_dir = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))

# Each rembg model maps to a specific .onnx filename; check common ones
model_files = list(u2net_dir.glob(f"{model}*.onnx"))
if model_files:
    size_mb = sum(f.stat().st_size for f in model_files) / 1024 / 1024
    print(f"✓ {model} already cached ({size_mb:.0f} MB)")
else:
    print(f"Downloading {model} via rembg...")
    from rembg import new_session
    new_session(model)
    print(f"✓ {model} downloaded to {u2net_dir}")
EOF

# ── MediaPipe face detector model ────────────────────────────────────────────
FACE_MODEL_FILE="$U2NET_DIR/blaze_face_short_range.tflite"
FACE_MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

if [ -f "$FACE_MODEL_FILE" ]; then
    size_kb=$(du -k "$FACE_MODEL_FILE" | cut -f1)
    echo "✓ blaze_face_short_range.tflite already cached (${size_kb} KB)"
else
    echo "Downloading blaze_face_short_range.tflite..."
    curl -fsSL --max-time 3600 -o "$FACE_MODEL_FILE" "$FACE_MODEL_URL"
    echo "✓ blaze_face_short_range.tflite downloaded to $U2NET_DIR"
fi

echo ""
echo "All models ready. You can now run: python main.py"
