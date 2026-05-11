from src.types.index import PaperMargins
from src.utils.crop_area_calculation import FaceBox, calculate_initial_crop_area
from src.utils.dpi_calculation import calculate_dpi
from src.utils.layout_calculation import PhotoSize, calculate_layout

# ── crop_area_calculation ──────────────────────────────────────────────────────


class TestCalculateInitialCropArea:
    def test_centred_face_fits_within_image(self) -> None:
        face = FaceBox(x=400, y=300, width=100, height=120)
        crop = calculate_initial_crop_area(face, 1.0, 1000, 1000)
        assert crop.x >= 0
        assert crop.y >= 0
        assert crop.x + crop.width <= 1000
        assert crop.y + crop.height <= 1000

    def test_aspect_ratio_preserved(self) -> None:
        face = FaceBox(x=400, y=300, width=100, height=120)
        ratio = 25 / 35
        crop = calculate_initial_crop_area(face, ratio, 1000, 1000)
        assert abs(crop.width / crop.height - ratio) < 1e-6

    def test_face_near_edge_clamps_crop(self) -> None:
        # Face in the top-left corner
        face = FaceBox(x=0, y=0, width=50, height=60)
        crop = calculate_initial_crop_area(face, 1.0, 500, 500)
        assert crop.x >= 0
        assert crop.y >= 0

    def test_crop_is_centred_horizontally_on_face_centre(self) -> None:
        # Horizontal axis is always symmetric — crop centre x == face centre x
        face = FaceBox(x=450, y=450, width=100, height=100)
        crop = calculate_initial_crop_area(face, 1.0, 1000, 1000)
        crop_center_x = crop.x + crop.width / 2
        assert abs(crop_center_x - 500) < 1e-6

    def test_crop_vertical_margins_are_asymmetric(self) -> None:
        # vertical_above = 1.0 * fh  (forehead/hair headroom)
        # vertical_below = 0.6 * fh  (neck/shoulder room)
        # The crop top should be face.y - 1.0*fh, bottom should be face.y+fh + 0.6*fh
        face = FaceBox(x=400, y=400, width=100, height=100)
        crop = calculate_initial_crop_area(face, 1.0, 2000, 2000)
        space_above = face.y - crop.y  # should be ≈ 1.0 * fh
        space_below = (crop.y + crop.height) - (
            face.y + face.height
        )  # should be ≈ 0.6 * fh
        # Allow tolerance of 5% of face height due to aspect-ratio adjustment
        assert (
            abs(space_above - 100.0) < 5.0
        ), f"space_above={space_above}, expected ≈100"
        assert abs(space_below - 60.0) < 5.0, f"space_below={space_below}, expected ≈60"

    def test_zero_size_crop_not_produced_when_face_at_edge(self) -> None:
        # Face with center at x=0 — must not produce zero-width or zero-height crop
        face = FaceBox(x=0, y=100, width=10, height=10)
        crop = calculate_initial_crop_area(face, 1.0, 500, 500)
        assert crop.width > 0
        assert crop.height > 0


# ── dpi_calculation ────────────────────────────────────────────────────────────


class TestCalculateDPI:
    def test_300dpi_at_25x35mm(self) -> None:
        # 25 mm / 25.4 mm/inch * 300 dpi ≈ 295 px wide
        width_px = round(25 / 25.4 * 300)
        height_px = round(35 / 25.4 * 300)
        result = calculate_dpi(width_px, height_px, 25, 35)
        assert abs(result.width_dpi - 300) < 1
        assert abs(result.height_dpi - 300) < 1
        assert result.min_dpi == min(result.width_dpi, result.height_dpi)

    def test_min_dpi_is_limiting_axis(self) -> None:
        result = calculate_dpi(100, 200, 10, 10)
        assert result.min_dpi == result.width_dpi  # width is the bottleneck


# ── layout_calculation ─────────────────────────────────────────────────────────


class TestCalculateLayout:
    def test_6inch_layout_produces_multiple_photos(self) -> None:
        layout = calculate_layout("6-inch", PhotoSize(25, 35), dpi=300)
        assert layout.photos_per_row >= 1
        assert layout.photos_per_column >= 1
        assert layout.total_photos == layout.photos_per_row * layout.photos_per_column

    def test_a4_layout_produces_more_photos_than_6inch(self) -> None:
        photo = PhotoSize(25, 35)
        a4 = calculate_layout("a4", photo, dpi=300)
        inch6 = calculate_layout("6-inch", photo, dpi=300)
        assert a4.total_photos >= inch6.total_photos

    def test_margins_shrink_printable_area(self) -> None:
        margins = PaperMargins(top=5, bottom=5, left=5, right=5)
        with_margins = calculate_layout(
            "a4", PhotoSize(25, 35), dpi=300, margins=margins
        )
        without_margins = calculate_layout("a4", PhotoSize(25, 35), dpi=300)
        # margins reduce the printable area, so we can fit ≤ photos
        assert with_margins.total_photos <= without_margins.total_photos

    def test_paper_pixel_dimensions_match_dpi(self) -> None:
        # 6-inch = 101.6 × 152.4 mm; at 300 dpi:
        # 101.6 / 25.4 * 300 = 1200 px,  152.4 / 25.4 * 300 = 1800 px
        layout = calculate_layout("6-inch", PhotoSize(25, 35), dpi=300)
        assert layout.paper_width_px == 1200
        assert layout.paper_height_px == 1800


# ── print_layout_service ──────────────────────────────────────────────────────


class TestPrintLayoutServiceDpi:
    """generate_print_layout must embed the requested DPI in the JPEG output."""

    def test_output_png_has_300_dpi(self) -> None:
        from io import BytesIO

        from PIL import Image

        from src.services.print_layout_service import generate_print_layout
        from src.utils.layout_calculation import PhotoSize

        photo_img = Image.new("RGB", (295, 413), color=(200, 180, 160))
        photo_buf = BytesIO()
        photo_img.save(photo_buf, format="PNG")
        photo_data = photo_buf.getvalue()

        result = generate_print_layout(
            photo_data=photo_data,
            photo_size=PhotoSize(25, 35),
            paper_type="6-inch",
            dpi=300,
        )

        out_img = Image.open(BytesIO(result))
        dpi = out_img.info.get("dpi")
        assert dpi is not None, "Layout PNG must contain DPI metadata"
        assert abs(dpi[0] - 300) < 1, f"Expected 300 DPI, got {dpi[0]}"
        assert abs(dpi[1] - 300) < 1, f"Expected 300 DPI, got {dpi[1]}"
