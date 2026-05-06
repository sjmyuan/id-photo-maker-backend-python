from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image, UnidentifiedImageError

VALID_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@dataclass
class ValidationResult:
    is_valid: bool
    file_size: int
    needs_scaling: bool
    dimensions: tuple[int, int]  # (width, height)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_image_buffer(data: bytes, mime_type: str) -> ValidationResult:
    """Validate an image buffer for MIME type, readability, and size."""
    if mime_type not in VALID_MIME_TYPES:
        return ValidationResult(
            is_valid=False,
            file_size=len(data),
            needs_scaling=False,
            dimensions=(0, 0),
            errors=["Invalid file type. Only JPEG, PNG, and WebP are supported."],
        )

    warnings: list[str] = []
    needs_scaling = len(data) > MAX_FILE_SIZE
    if needs_scaling:
        warnings.append(
            "File size exceeds 10 MB. Image will be automatically scaled down."
        )

    try:
        img = Image.open(BytesIO(data))
        width, height = img.size  # read dimensions before verify closes the handle
        img.verify()
        return ValidationResult(
            is_valid=True,
            file_size=len(data),
            needs_scaling=needs_scaling,
            dimensions=(width, height),
            warnings=warnings,
        )
    except (UnidentifiedImageError, Exception):
        return ValidationResult(
            is_valid=False,
            file_size=len(data),
            needs_scaling=False,
            dimensions=(0, 0),
            errors=["Failed to read image file."],
        )
