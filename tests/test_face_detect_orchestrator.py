"""Tests for face_detect_orchestrator — normalised bbox contract."""

from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from src.services.face_detect_orchestrator import (
    NormalisedFaceBox,
    detect_face_in_image,
)
from src.utils.crop_area_calculation import FaceBox


def _make_image_bytes(width: int = 200, height: int = 200) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 180, 160))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_mock_face_detection_model() -> MagicMock:
    model = MagicMock()
    return model


class TestNormalisedFaceBox:
    def test_values_are_in_0_to_1_range(self) -> None:
        box = NormalisedFaceBox(x=0.1, y=0.2, width=0.3, height=0.4)
        assert 0.0 <= box.x <= 1.0
        assert 0.0 <= box.y <= 1.0
        assert 0.0 <= box.width <= 1.0
        assert 0.0 <= box.height <= 1.0


class TestDetectFaceInImage:
    def test_returns_normalised_face_box_on_success(self) -> None:
        """detect_face_in_image must return NormalisedFaceBox with values in 0–1."""
        image_data = _make_image_bytes(400, 600)
        model = _make_mock_face_detection_model()

        # Face at absolute pixel (100, 150, w=80, h=100) in a 400×600 image
        face_px = FaceBox(x=100, y=150, width=80, height=100)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[face_px]),
        ):
            result = detect_face_in_image(image_data, "image/jpeg", model)

        assert result.errors == []
        assert result.face is not None
        # Normalised: x=100/400=0.25, y=150/600=0.25, w=80/400=0.2, h=100/600≈0.1667
        assert abs(result.face.x - 0.25) < 1e-6
        assert abs(result.face.y - 0.25) < 1e-6
        assert abs(result.face.width - 0.20) < 1e-6
        assert abs(result.face.height - (100 / 600)) < 1e-6

    def test_normalised_values_never_exceed_1(self) -> None:
        """Even if face bbox is near edge, normalised values stay ≤ 1."""
        image_data = _make_image_bytes(100, 100)
        model = _make_mock_face_detection_model()
        face_px = FaceBox(x=0, y=0, width=100, height=100)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[face_px]),
        ):
            result = detect_face_in_image(image_data, "image/jpeg", model)

        assert result.face is not None
        assert result.face.x == 0.0
        assert result.face.y == 0.0
        assert result.face.width == 1.0
        assert result.face.height == 1.0

    def test_no_face_returns_error(self) -> None:
        image_data = _make_image_bytes()
        model = _make_mock_face_detection_model()

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[]),
        ):
            result = detect_face_in_image(image_data, "image/jpeg", model)

        assert result.face is None
        assert any(e.type == "face-detection" for e in result.errors)

    def test_multiple_faces_returns_error(self) -> None:
        image_data = _make_image_bytes()
        model = _make_mock_face_detection_model()
        faces = [
            FaceBox(x=10, y=10, width=30, height=30),
            FaceBox(x=100, y=10, width=30, height=30),
        ]

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=faces),
        ):
            result = detect_face_in_image(image_data, "image/jpeg", model)

        assert result.face is None
        assert any(e.type == "face-detection" for e in result.errors)

    def test_invalid_mime_type_returns_validation_error(self) -> None:
        image_data = _make_image_bytes()
        model = _make_mock_face_detection_model()
        result = detect_face_in_image(image_data, "image/gif", model)
        assert any(e.type == "validation" for e in result.errors)

    def test_image_width_height_not_returned(self) -> None:
        """/detect no longer needs to return absolute image dimensions."""
        image_data = _make_image_bytes(320, 480)
        model = _make_mock_face_detection_model()
        face_px = FaceBox(x=80, y=120, width=64, height=80)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[face_px]),
        ):
            result = detect_face_in_image(image_data, "image/jpeg", model)

        # image_width / image_height fields should be gone from the result
        assert not hasattr(result, "image_width")
        assert not hasattr(result, "image_height")
