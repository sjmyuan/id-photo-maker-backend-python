from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass
class DPIResult:
    width_dpi: float
    height_dpi: float
    min_dpi: float


def calculate_dpi(
    width_px: float,
    height_px: float,
    width_mm: float,
    height_mm: float,
) -> DPIResult:
    """Calculate DPI based on crop area pixel dimensions and physical size in mm."""
    width_dpi = (width_px / width_mm) * MM_PER_INCH
    height_dpi = (height_px / height_mm) * MM_PER_INCH
    return DPIResult(
        width_dpi=width_dpi, height_dpi=height_dpi, min_dpi=min(width_dpi, height_dpi)
    )
