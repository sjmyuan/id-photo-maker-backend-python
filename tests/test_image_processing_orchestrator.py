"""Tests for image_processing_orchestrator — pre-cropped image contract."""

from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from src.services.image_processing_orchestrator import NormalisedCropArea, process_image
from src.types.index import SIZE_OPTIONS_BY_ID


def _make_image_bytes(width: int = 295, height: int = 413) -> bytes:
    """Default size ≈ 25×35 mm at 300 DPI (295×413 px)."""
    img = Image.new("RGB", (width, height), color=(180, 160, 140))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_png_bytes(width: int = 295, height: int = 413) -> bytes:
    return _make_image_bytes(width, height)


def _make_pil_image(width: int = 295, height: int = 413) -> Image.Image:
    return Image.new("RGBA", (width, height), color=(180, 160, 140, 255))


SELECTED_SIZE = SIZE_OPTIONS_BY_ID["1-inch"]  # 25×35 mm


class TestProcessImagePreCropped:
    """process_image now receives an already-cropped image — no face detection, no crop step."""

    def test_no_face_detection_called(self) -> None:
        """process_image must NOT import or call detect_faces_in_buffer."""
        import src.services.image_processing_orchestrator as mod

        assert not hasattr(
            mod, "detect_faces_in_buffer"
        ), "detect_faces_in_buffer should not be present in the orchestrator module"

    def test_no_face_detection_model_parameter_needed(self) -> None:
        """process_image signature should not require face_detection_model."""
        import inspect

        sig = inspect.signature(process_image)
        assert "face_detection_model" not in sig.parameters

    def test_returns_success_with_b64_id_photo(self) -> None:
        image_data = _make_png_bytes()
        u2net = MagicMock()
        _mod = "src.services.image_processing_orchestrator"

        with (
            patch(f"{_mod}.remove_background", return_value=_make_pil_image()),
            patch(
                f"{_mod}.generate_exact_crop_from_image", return_value=_make_pil_image()
            ),
        ):
            result = process_image(
                image_data=image_data,
                mime_type="image/png",
                selected_size=SELECTED_SIZE,
                background_color="#0000FF",
                u2net_model=u2net,
            )

        assert result.errors == []
        assert result.result is not None
        assert len(result.result.id_photo_b64) > 0

    def test_dpi_warning_when_below_required(self) -> None:
        """Low-res crop → DPI warning (not error), processing still continues."""
        # 50×70 px for 25×35 mm ≈ 50.8 DPI — well below 300
        image_data = _make_png_bytes(width=50, height=70)
        u2net = MagicMock()
        _mod = "src.services.image_processing_orchestrator"

        with (
            patch(f"{_mod}.remove_background", return_value=_make_pil_image()),
            patch(
                f"{_mod}.generate_exact_crop_from_image", return_value=_make_pil_image()
            ),
        ):
            result = process_image(
                image_data=image_data,
                mime_type="image/png",
                selected_size=SELECTED_SIZE,
                background_color="#FFFFFF",
                u2net_model=u2net,
                required_dpi=300,
            )

        # Should NOT be an error — just a warning; result is still produced
        assert result.errors == []
        assert result.result is not None
        assert any("DPI" in w for w in result.warnings)

    def test_invalid_mime_returns_validation_error(self) -> None:
        image_data = _make_png_bytes()
        u2net = MagicMock()

        result = process_image(
            image_data=image_data,
            mime_type="image/gif",
            selected_size=SELECTED_SIZE,
            background_color="#FFFFFF",
            u2net_model=u2net,
        )

        assert any(e.type == "validation" for e in result.errors)
        assert result.result is None


class TestProcessImageWithNormalisedCropArea:
    """When normalised_crop_area is supplied, process_image crops directly and skips
    the calculate_initial_crop_area step."""

    def test_calculate_initial_crop_area_not_called_with_crop_area(self) -> None:
        """When normalised_crop_area is provided, calculate_initial_crop_area must NOT run."""
        image_data = _make_png_bytes(400, 600)
        u2net = MagicMock()
        _mod = "src.services.image_processing_orchestrator"

        # Crop area covering the centre of a 400×600 image (normalised)
        crop = NormalisedCropArea(x=0.1, y=0.1, width=0.8, height=0.8)

        with (
            patch(f"{_mod}.calculate_initial_crop_area") as mock_calc,
            patch(f"{_mod}.remove_background", return_value=_make_pil_image()),
            patch(
                f"{_mod}.generate_exact_crop_from_image", return_value=_make_pil_image()
            ),
        ):
            process_image(
                image_data=image_data,
                mime_type="image/png",
                selected_size=SELECTED_SIZE,
                background_color="#0000FF",
                u2net_model=u2net,
                normalised_crop_area=crop,
            )

        mock_calc.assert_not_called()

    def test_returns_success_with_normalised_crop_area(self) -> None:
        image_data = _make_png_bytes(400, 600)
        u2net = MagicMock()
        _mod = "src.services.image_processing_orchestrator"
        crop = NormalisedCropArea(x=0.1, y=0.1, width=0.8, height=0.8)

        with (
            patch(f"{_mod}.remove_background", return_value=_make_pil_image()),
            patch(
                f"{_mod}.generate_exact_crop_from_image", return_value=_make_pil_image()
            ),
        ):
            result = process_image(
                image_data=image_data,
                mime_type="image/png",
                selected_size=SELECTED_SIZE,
                background_color="#0000FF",
                u2net_model=u2net,
                normalised_crop_area=crop,
            )

        assert result.errors == []
        assert result.result is not None

    def test_normalised_face_still_works_alongside_crop_area(self) -> None:
        """normalised_face should be ignored when normalised_crop_area is provided."""
        from src.services.image_processing_orchestrator import NormalisedFace

        image_data = _make_png_bytes(400, 600)
        u2net = MagicMock()
        _mod = "src.services.image_processing_orchestrator"
        crop = NormalisedCropArea(x=0.1, y=0.1, width=0.8, height=0.8)
        face = NormalisedFace(x=0.3, y=0.3, width=0.2, height=0.2)

        with (
            patch(f"{_mod}.calculate_initial_crop_area") as mock_calc,
            patch(f"{_mod}.remove_background", return_value=_make_pil_image()),
            patch(
                f"{_mod}.generate_exact_crop_from_image", return_value=_make_pil_image()
            ),
        ):
            result = process_image(
                image_data=image_data,
                mime_type="image/png",
                selected_size=SELECTED_SIZE,
                background_color="#0000FF",
                u2net_model=u2net,
                normalised_face=face,
                normalised_crop_area=crop,
            )

        # crop_area takes precedence; face calculation must NOT be triggered
        mock_calc.assert_not_called()
        assert result.errors == []


class TestOutputDpiMetadata:
    """The final JPEG returned by process_image must embed 300 DPI metadata."""

    def test_output_png_has_300_dpi(self) -> None:
        import base64

        image_data = _make_png_bytes(295, 413)
        u2net = MagicMock()
        _mod = "src.services.image_processing_orchestrator"

        with (
            patch(f"{_mod}.remove_background", return_value=_make_pil_image(295, 413)),
            patch(
                f"{_mod}.generate_exact_crop_from_image",
                return_value=_make_pil_image(295, 413),
            ),
        ):
            result = process_image(
                image_data=image_data,
                mime_type="image/png",
                selected_size=SELECTED_SIZE,
                background_color="#FFFFFF",
                u2net_model=u2net,
                required_dpi=300,
            )

        assert result.result is not None
        img_bytes = base64.b64decode(result.result.id_photo_b64)
        img = Image.open(BytesIO(img_bytes))
        dpi = img.info.get("dpi")
        assert dpi is not None, "PNG output must contain DPI metadata"
        assert abs(dpi[0] - 300) < 1, f"Expected 300 DPI, got {dpi[0]}"
        assert abs(dpi[1] - 300) < 1, f"Expected 300 DPI, got {dpi[1]}"
