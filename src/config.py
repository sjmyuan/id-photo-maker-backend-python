import os

from dotenv import load_dotenv

load_dotenv()


def _parse_cors(raw: str | None) -> list[str] | str:
    if not raw:
        return "*"
    return [o.strip() for o in raw.split(",")]


PORT: int = int(os.getenv("PORT", "3000"))

CORS_ORIGIN: list[str] | str = _parse_cors(os.getenv("CORS_ORIGIN"))

# Soft threshold: images above this are scaled down before processing
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024

# Hard upload limit: requests larger than this are rejected with 413
MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024
