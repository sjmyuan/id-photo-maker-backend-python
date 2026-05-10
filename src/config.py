import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _parse_cors(raw: str | None) -> list[str] | str:
    if not raw:
        return "*"
    return [o.strip() for o in raw.split(",")]


PORT: int = int(os.getenv("PORT", "3000"))

# Directory where AI model files (rembg ONNX + mediapipe tflite) are stored.
# Defaults to ~/.u2net (same as rembg); overridden to /app/models in Docker via U2NET_HOME.
MODELS_DIR: Path = Path(os.getenv("U2NET_HOME", str(Path.home() / ".u2net")))

CORS_ORIGIN: list[str] | str = _parse_cors(os.getenv("CORS_ORIGIN"))

# Background removal model name (any rembg-supported model name)
# birefnet-portrait is recommended for ID photos; u2net is the smaller fallback
BG_REMOVAL_MODEL: str = os.getenv("BG_REMOVAL_MODEL", "u2net_human_seg")

# Soft threshold: images above this are scaled down before processing
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024

# Hard upload limit: requests larger than this are rejected with 413
MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024
