from dataclasses import dataclass
from typing import Literal

from src.types.index import PaperMargins

MM_PER_INCH = 25.4

PaperTypeId = Literal["6-inch", "a4"]

# 6-inch photo paper: 4" × 6" (101.6 mm × 152.4 mm)
# A4 paper: 210 mm × 297 mm
PAPER_SIZES: dict[str, tuple[float, float]] = {
    "6-inch": (101.6, 152.4),
    "a4": (210.0, 297.0),
}


@dataclass
class PhotoSize:
    width_mm: float
    height_mm: float


@dataclass
class PrinterMarginsPx:
    top: float
    bottom: float
    left: float
    right: float


@dataclass
class LayoutResult:
    paper_type: str
    paper_width_px: int
    paper_height_px: int
    photo_width_px: float
    photo_height_px: float
    photos_per_row: int
    photos_per_column: int
    total_photos: int
    horizontal_spacing_px: float
    vertical_spacing_px: float
    margin_left_px: float
    margin_top_px: float
    printer_margins: PrinterMarginsPx | None


def mm_to_pixels(mm: float, dpi: float) -> float:
    return (mm * dpi) / MM_PER_INCH


def calculate_layout(
    paper_type_id: PaperTypeId,
    photo_size: PhotoSize,
    dpi: float = 300,
    margins: PaperMargins | None = None,
) -> LayoutResult:
    paper_w_mm, paper_h_mm = PAPER_SIZES[paper_type_id]

    paper_w_px = round(mm_to_pixels(paper_w_mm, dpi))
    paper_h_px = round(mm_to_pixels(paper_h_mm, dpi))

    photo_w_px = mm_to_pixels(photo_size.width_mm, dpi)
    photo_h_px = mm_to_pixels(photo_size.height_mm, dpi)

    if margins:
        pm = PrinterMarginsPx(
            top=mm_to_pixels(margins.top, dpi),
            bottom=mm_to_pixels(margins.bottom, dpi),
            left=mm_to_pixels(margins.left, dpi),
            right=mm_to_pixels(margins.right, dpi),
        )
    else:
        pm = PrinterMarginsPx(top=0, bottom=0, left=0, right=0)

    printable_w = paper_w_px - pm.left - pm.right
    printable_h = paper_h_px - pm.top - pm.bottom

    min_spacing = mm_to_pixels(5, dpi)

    photos_per_row = max(1, int(printable_w // (photo_w_px + min_spacing)))
    photos_per_column = max(1, int(printable_h // (photo_h_px + min_spacing)))

    h_spacing = (
        (printable_w - photos_per_row * photo_w_px) / (photos_per_row + 1)
        if photos_per_row > 1
        else 0.0
    )
    v_spacing = (
        (printable_h - photos_per_column * photo_h_px) / (photos_per_column + 1)
        if photos_per_column > 1
        else 0.0
    )

    layout_margin_left = (
        (printable_w - photo_w_px) / 2 if photos_per_row == 1 else h_spacing
    )
    layout_margin_top = (
        (printable_h - photo_h_px) / 2 if photos_per_column == 1 else v_spacing
    )

    margin_left_px = pm.left + layout_margin_left
    margin_top_px = pm.top + layout_margin_top

    return LayoutResult(
        paper_type=paper_type_id,
        paper_width_px=paper_w_px,
        paper_height_px=paper_h_px,
        photo_width_px=photo_w_px,
        photo_height_px=photo_h_px,
        photos_per_row=photos_per_row,
        photos_per_column=photos_per_column,
        total_photos=photos_per_row * photos_per_column,
        horizontal_spacing_px=h_spacing,
        vertical_spacing_px=v_spacing,
        margin_left_px=margin_left_px,
        margin_top_px=margin_top_px,
        printer_margins=pm if margins else None,
    )
