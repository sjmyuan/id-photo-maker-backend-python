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


def generate_exact_crop_from_image(
    img: Image.Image,
    crop_area: CropArea,
    width_mm: float,
    height_mm: float,
    dpi: float,
) -> Image.Image:
    """Crop a region and resize to exact physical dimensions at the given DPI.
    Accepts and returns PIL Image objects to avoid PNG encode/decode round-trips.
    """
    target_w, target_h = _target_pixel_dimensions(width_mm, height_mm, dpi)

    left = max(0, round(crop_area.x))
    top = max(0, round(crop_area.y))
    right = left + round(crop_area.width)
    bottom = top + round(crop_area.height)

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((target_w, target_h), Image.LANCZOS)
