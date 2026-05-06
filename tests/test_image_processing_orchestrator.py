"""Tests for image_processing_orchestrator — pre-cropped image contract."""

from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from src.services.image_processing_orchestrator import process_image
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
