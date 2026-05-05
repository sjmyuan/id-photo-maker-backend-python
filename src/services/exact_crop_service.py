from io import BytesIO

from PIL import Image

from src.utils.crop_area_calculation import CropArea

MM_PER_INCH = 25.4


def _target_pixel_dimensions(
    width_mm: float, height_mm: float, dpi: float
) -> tuple[int, int]:
    return (
        round((width_mm / MM_PER_INCH) * dpi),
        round((height_mm / MM_PER_INCH) * dpi),
    )


def generate_exact_crop(
    source_data: bytes,
    crop_area: CropArea,
    width_mm: float,
    height_mm: float,
    dpi: float,
) -> bytes:
    """Crop a region and resize to exact physical dimensions at the given DPI."""
    target_w, target_h = _target_pixel_dimensions(width_mm, height_mm, dpi)

    img = Image.open(BytesIO(source_data))
    left = max(0, round(crop_area.x))
    top = max(0, round(crop_area.y))
    right = left + round(crop_area.width)
    bottom = top + round(crop_area.height)

    cropped = img.crop((left, top, right, bottom))
    resized = cropped.resize((target_w, target_h), Image.LANCZOS)

    buf = BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()
