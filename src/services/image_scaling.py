import math
from io import BytesIO

from PIL import Image

TARGET_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def scale_image_to_target(data: bytes) -> tuple[bytes, tuple[int, int]]:
    """Scale an image down to fit within 10 MB.

    Returns a tuple of ``(image_bytes, (width, height))``.  When no scaling is
    needed the original bytes are returned unchanged alongside the original
    dimensions, avoiding a second ``Image.open`` call in the caller.
    """
    img = Image.open(BytesIO(data))
    orig_w, orig_h = img.size

    if orig_w == 0 or orig_h == 0:
        raise ValueError("Cannot determine image dimensions for scaling.")

    if len(data) <= TARGET_MAX_SIZE_BYTES:
        return data, (orig_w, orig_h)

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
    return buf.getvalue(), (new_w, new_h)
