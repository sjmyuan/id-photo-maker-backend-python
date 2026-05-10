from io import BytesIO

from PIL import Image

from src.types.index import PaperMargins
from src.utils.layout_calculation import PhotoSize, calculate_layout


def generate_print_layout(
    photo_data: bytes,
    photo_size: PhotoSize,
    paper_type: str,
    dpi: float = 300,
    margins: PaperMargins | None = None,
) -> bytes:
    """Generate a high-resolution print-ready layout PNG with multiple ID photos in a grid."""
    layout = calculate_layout(paper_type, photo_size, dpi, margins)  # type: ignore[arg-type]

    offset_x = layout.printer_margins.left if layout.printer_margins else 0
    offset_y = layout.printer_margins.top if layout.printer_margins else 0

    canvas_w = (
        layout.paper_width_px
        - layout.printer_margins.left
        - layout.printer_margins.right
        if layout.printer_margins
        else layout.paper_width_px
    )
    canvas_h = (
        layout.paper_height_px
        - layout.printer_margins.top
        - layout.printer_margins.bottom
        if layout.printer_margins
        else layout.paper_height_px
    )

    # Resize the source photo once to the target cell size
    photo_img = Image.open(BytesIO(photo_data))
    cell_w = round(layout.photo_width_px)
    cell_h = round(layout.photo_height_px)
    resized_photo = photo_img.resize((cell_w, cell_h), Image.LANCZOS)

    # White background canvas
    canvas = Image.new("RGB", (round(canvas_w), round(canvas_h)), color=(255, 255, 255))

    for row in range(layout.photos_per_column):
        for col in range(layout.photos_per_row):
            x = round(
                layout.margin_left_px
                - offset_x
                + col * (layout.photo_width_px + layout.horizontal_spacing_px)
            )
            y = round(
                layout.margin_top_px
                - offset_y
                + row * (layout.photo_height_px + layout.vertical_spacing_px)
            )
            canvas.paste(resized_photo, (x, y))

    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
