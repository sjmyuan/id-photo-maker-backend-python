import math
from io import BytesIO

from PIL import Image

TARGET_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def scale_image_to_target(data: bytes) -> bytes:
    """Scale an image down to fit within 10 MB. Returns data unchanged if already small enough."""
    if len(data) <= TARGET_MAX_SIZE_BYTES:
        return data

    img = Image.open(BytesIO(data))
    orig_w, orig_h = img.size

    if orig_w == 0 or orig_h == 0:
        raise ValueError("Cannot determine image dimensions for scaling.")

    size_ratio = TARGET_MAX_SIZE_BYTES / len(data)
    scale_factor = (
        math.sqrt(size_ratio) * 0.9
    )  # slightly aggressive to stay under limit

    new_w = max(1, int(orig_w * scale_factor))
    new_h = max(1, int(orig_h * scale_factor))

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    buf = BytesIO()
    fmt = img.format or "JPEG"
    resized.save(buf, format=fmt)
    return buf.getvalue()
