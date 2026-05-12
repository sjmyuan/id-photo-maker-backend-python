"""Integration tests for the FastAPI server endpoints."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.server import Models, create_app
from src.types.index import BG_COLOR_OPTIONS, SIZE_OPTIONS
from src.utils.crop_area_calculation import FaceBox


@pytest.fixture()
def client() -> TestClient:
    models = Models(u2net=MagicMock(), face_detection=MagicMock())
    app = create_app(models)
    return TestClient(app)


# ── GET /api/options ──────────────────────────────────────────────────────────


class TestGetOptions:
    def test_returns_200(self, client: TestClient) -> None:
        res = client.get("/api/options")
        assert res.status_code == 200

    def test_response_has_sizes_and_colors(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        assert "sizes" in data
        assert "colors" in data

    def test_sizes_count_matches_size_options(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        assert len(data["sizes"]) == len(SIZE_OPTIONS)

    def test_colors_count_matches_bg_color_options(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        assert len(data["colors"]) == len(BG_COLOR_OPTIONS)

    def test_size_item_has_required_fields(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        size = data["sizes"][0]
        assert "id" in size
        assert "label" in size
        assert "labelZh" in size
        assert "dims" in size
        assert "widthMm" in size
        assert "heightMm" in size
        assert "aspectRatio" in size

    def test_color_item_has_required_fields(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        color = data["colors"][0]
        assert "label" in color
        assert "value" in color

    def test_size_ids_match_known_ids(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        returned_ids = {s["id"] for s in data["sizes"]}
        expected_ids = {s.id for s in SIZE_OPTIONS}
        assert returned_ids == expected_ids

    def test_size_aspect_ratio_matches_dimensions(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        for size in data["sizes"]:
            expected_ratio = size["widthMm"] / size["heightMm"]
            assert abs(size["aspectRatio"] - expected_ratio) < 1e-6

    def test_color_values_are_valid_hex(self, client: TestClient) -> None:
        import re

        data = client.get("/api/options").json()
        for color in data["colors"]:
            assert re.fullmatch(
                r"#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})", color["value"]
            ), f"Invalid hex colour: {color['value']}"

    def test_first_color_is_blue(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        assert data["colors"][0]["value"] == "#0000FF"


# ── POST /api/detect (with sizeId) ───────────────────────────────────────────


def _make_jpeg_bytes(width: int = 400, height: int = 600) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 180, 160))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestDetectWithSizeId:
    """When sizeId is supplied, /api/detect returns cropArea and dpiCheck."""

    # 800×1200 image with face 200×250: crop_height ≈ 2.6×250 = 650 px → ~471 DPI → sufficient
    _LARGE_FACE = FaceBox(x=200, y=350, width=200, height=250)

    def test_returns_crop_area_when_size_id_provided(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes(800, 1200)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[self._LARGE_FACE]),
        ):
            res = client.post(
                "/api/detect",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"sizeId": "1-inch"},
            )

        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "cropArea" in data
        ca = data["cropArea"]
        assert "x" in ca and "y" in ca and "width" in ca and "height" in ca

    def test_crop_area_values_in_valid_range(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes(800, 1200)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[self._LARGE_FACE]),
        ):
            res = client.post(
                "/api/detect",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"sizeId": "1-inch"},
            )

        ca = res.json()["cropArea"]
        assert 0.0 <= ca["x"] <= 1.0
        assert 0.0 <= ca["y"] <= 1.0
        assert 0.0 < ca["width"] <= 1.0
        assert 0.0 < ca["height"] <= 1.0

    def test_returns_dpi_check_when_size_id_provided(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes(800, 1200)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[self._LARGE_FACE]),
        ):
            res = client.post(
                "/api/detect",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"sizeId": "1-inch"},
            )

        assert res.status_code == 200
        data = res.json()
        assert "dpiCheck" in data
        assert data["dpiCheck"]["sufficient"] is True

    def test_returns_422_when_dpi_insufficient(self, client: TestClient) -> None:
        # 400×600 image, face 80×100 → crop_height ≈ 260 px → ~188 DPI → insufficient
        image_bytes = _make_jpeg_bytes(400, 600)
        small_face = FaceBox(x=100, y=150, width=80, height=100)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[small_face]),
        ):
            res = client.post(
                "/api/detect",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"sizeId": "1-inch"},
            )

        assert res.status_code == 422
        data = res.json()
        assert data["success"] is False
        assert data["errors"][0]["type"] == "low-dpi"

    def test_crop_area_is_none_without_size_id(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes()
        face_px = FaceBox(x=100, y=150, width=80, height=100)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[face_px]),
        ):
            res = client.post(
                "/api/detect",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            )

        assert res.status_code == 200
        data = res.json()
        assert data.get("cropArea") is None
        assert data.get("dpiCheck") is None

    def test_invalid_size_id_returns_400(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes()
        face_px = FaceBox(x=100, y=150, width=80, height=100)

        with patch(
            "src.services.face_detect_orchestrator.detect_faces_in_buffer",
            return_value=MagicMock(faces=[face_px]),
        ):
            res = client.post(
                "/api/detect",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"sizeId": "invalid-size"},
            )

        assert res.status_code == 400
        data = res.json()
        assert data["success"] is False


# ── POST /api/process (with cropArea) ────────────────────────────────────────


class TestProcessWithCropArea:
    """When cropX/cropY/cropW/cropH are supplied, /api/process uses them directly."""

    def test_accepts_crop_area_fields(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes(400, 600)

        mock_result = MagicMock()
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.result = MagicMock(id_photo_b64="AAAA")

        with patch("src.server.process_image", return_value=mock_result) as mock_proc:
            res = client.post(
                "/api/process",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={
                    "sizeId": "1-inch",
                    "backgroundColor": "#0000FF",
                    "cropX": "0.1",
                    "cropY": "0.1",
                    "cropW": "0.8",
                    "cropH": "0.8",
                },
            )

        assert res.status_code == 200
        # Verify normalised_crop_area was passed (not normalised_face)
        call_kwargs = mock_proc.call_args.kwargs
        assert call_kwargs.get("normalised_crop_area") is not None
        assert call_kwargs.get("normalised_face") is None

    def test_partial_crop_fields_returns_400(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes()

        res = client.post(
            "/api/process",
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            data={
                "sizeId": "1-inch",
                "backgroundColor": "#0000FF",
                "cropX": "0.1",
                "cropY": "0.1",
                # cropW and cropH missing
            },
        )

        assert res.status_code == 400
        assert res.json()["success"] is False

    def test_crop_area_out_of_range_returns_400(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes()

        res = client.post(
            "/api/process",
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            data={
                "sizeId": "1-inch",
                "backgroundColor": "#0000FF",
                "cropX": "1.5",  # invalid: > 1.0
                "cropY": "0.1",
                "cropW": "0.5",
                "cropH": "0.5",
            },
        )

        assert res.status_code == 400
        assert res.json()["success"] is False


# ── POST /api/layout ──────────────────────────────────────────────────────────


class TestLayout:
    """Tests for the /api/layout endpoint."""

    def test_6inch_returns_200_with_print_layout(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes()

        with patch(
            "src.server.generate_print_layout",
            return_value=image_bytes,
        ):
            res = client.post(
                "/api/layout",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"sizeId": "1-inch", "paperType": "6-inch"},
            )

        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "printLayout" in data
        assert len(data["printLayout"]) > 0

    def test_a4_returns_200_with_print_layout(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes()

        with patch(
            "src.server.generate_print_layout",
            return_value=image_bytes,
        ):
            res = client.post(
                "/api/layout",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"sizeId": "1-inch", "paperType": "a4"},
            )

        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_invalid_paper_type_returns_400(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes()

        res = client.post(
            "/api/layout",
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            data={"sizeId": "1-inch", "paperType": "letter"},
        )

        assert res.status_code == 400
        data = res.json()
        assert data["success"] is False
        assert data["errors"][0]["type"] == "validation"

    def test_invalid_size_id_returns_400(self, client: TestClient) -> None:
        image_bytes = _make_jpeg_bytes()

        res = client.post(
            "/api/layout",
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            data={"sizeId": "invalid-size", "paperType": "6-inch"},
        )

        assert res.status_code == 400
        data = res.json()
        assert data["success"] is False
        assert data["errors"][0]["type"] == "validation"

    def test_invalid_mime_type_returns_415(self, client: TestClient) -> None:
        res = client.post(
            "/api/layout",
            files={"image": ("file.gif", b"GIF89a", "image/gif")},
            data={"sizeId": "1-inch", "paperType": "6-inch"},
        )

        assert res.status_code == 415
        assert res.json()["success"] is False

    def test_oversized_upload_returns_413(self, client: TestClient) -> None:
        from src import config as cfg

        oversized = b"x" * (cfg.MAX_UPLOAD_SIZE_BYTES + 1)

        res = client.post(
            "/api/layout",
            files={"image": ("photo.jpg", oversized, "image/jpeg")},
            data={"sizeId": "1-inch", "paperType": "6-inch"},
        )

        assert res.status_code == 413
        assert res.json()["success"] is False

    def test_print_layout_is_valid_base64(self, client: TestClient) -> None:
        import base64

        image_bytes = _make_jpeg_bytes()

        with patch(
            "src.server.generate_print_layout",
            return_value=image_bytes,
        ):
            res = client.post(
                "/api/layout",
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"sizeId": "1-inch", "paperType": "6-inch"},
            )

        assert res.status_code == 200
        layout_b64 = res.json()["printLayout"]
        decoded = base64.b64decode(layout_b64)
        assert len(decoded) > 0
