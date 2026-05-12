import base64
import re
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import config
from src.services.face_detect_orchestrator import detect_face_in_image
from src.services.face_detection_service import FaceDetectionModel
from src.services.image_processing_orchestrator import (
    NormalisedCropArea,
    NormalisedFace,
    process_image,
)
from src.services.print_layout_service import generate_print_layout
from src.services.u2net_service import U2NetModel
from src.types.index import BG_COLOR_OPTIONS, SIZE_OPTIONS, SIZE_OPTIONS_BY_ID
from src.utils.layout_calculation import PhotoSize

ALLOWED_PAPER_TYPES = {"6-inch", "a4"}

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _validate_upload(data: bytes, mime: str) -> JSONResponse | None:
    """Return an error JSONResponse for an invalid upload, or None when the upload is valid."""
    if len(data) > config.MAX_UPLOAD_SIZE_BYTES:
        max_mb = config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "errors": [
                    {
                        "type": "validation",
                        "message": f"File too large. Maximum allowed size is {max_mb} MB.",
                    }
                ],
            },
        )
    if mime not in ALLOWED_MIME_TYPES:
        return JSONResponse(
            status_code=415,
            content={
                "success": False,
                "errors": [
                    {
                        "type": "validation",
                        "message": "Invalid file type. Only JPEG, PNG, and WebP are supported.",
                    }
                ],
            },
        )
    return None


@dataclass
class Models:
    u2net: U2NetModel
    face_detection: FaceDetectionModel


def create_app(models: Models) -> FastAPI:
    app = FastAPI(title="ID Photo Maker")

    # CORS
    origins = config.CORS_ORIGIN if isinstance(config.CORS_ORIGIN, list) else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ── GET /api/options ─────────────────────────────────────────────────────

    @app.get("/api/options")
    async def options() -> JSONResponse:
        return JSONResponse(
            content={
                "sizes": [
                    {
                        "id": s.id,
                        "label": s.label,
                        "labelZh": s.label_zh,
                        "dims": s.dimensions,
                        "widthMm": s.physical_width,
                        "heightMm": s.physical_height,
                        "aspectRatio": s.physical_width / s.physical_height,
                    }
                    for s in SIZE_OPTIONS
                ],
                "colors": [
                    {"label": c.label, "value": c.value} for c in BG_COLOR_OPTIONS
                ],
            }
        )

    # ── POST /api/detect ──────────────────────────────────────────────────────

    @app.post("/api/detect")
    async def detect(
        image: UploadFile = File(...),
        sizeId: str | None = Form(default=None),
    ) -> JSONResponse:
        data = await image.read()

        mime = image.content_type or ""
        upload_error = _validate_upload(data, mime)
        if upload_error is not None:
            return upload_error

        # Validate sizeId when provided
        size_option = None
        if sizeId is not None:
            size_option = SIZE_OPTIONS_BY_ID.get(sizeId)
            if size_option is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "errors": [
                            {
                                "type": "validation",
                                "message": "Invalid sizeId. See /api/options for valid values.",
                            }
                        ],
                    },
                )

        result = detect_face_in_image(
            data, mime, models.face_detection, size_option=size_option
        )

        if result.errors:
            has_face_error = any(e.type == "face-detection" for e in result.errors)
            status = 422 if has_face_error else 400
            return JSONResponse(
                status_code=status,
                content={
                    "success": False,
                    "errors": [
                        {"type": e.type, "message": e.message} for e in result.errors
                    ],
                },
            )

        if result.dpi_check is not None and not result.dpi_check.sufficient:
            calc_dpi = round(result.dpi_check.dpi)
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "low-dpi",
                            "message": (
                                f"Image resolution is too low (approx. {calc_dpi} DPI). "
                                "Please upload a higher-resolution photo (at least 300 DPI)."
                            ),
                        }
                    ],
                },
            )

        face = result.face
        face_dict = (
            {"x": face.x, "y": face.y, "width": face.width, "height": face.height}
            if face
            else None
        )
        crop_area = result.crop_area
        crop_area_dict = (
            {
                "x": crop_area.x,
                "y": crop_area.y,
                "width": crop_area.width,
                "height": crop_area.height,
            }
            if crop_area
            else None
        )
        dpi_check = result.dpi_check
        dpi_check_dict = (
            {"dpi": dpi_check.dpi, "sufficient": dpi_check.sufficient}
            if dpi_check
            else None
        )
        return JSONResponse(
            content={
                "success": True,
                "face": face_dict,
                "cropArea": crop_area_dict,
                "dpiCheck": dpi_check_dict,
                "warnings": result.warnings,
            }
        )

    # ── POST /api/process ─────────────────────────────────────────────────────

    @app.post("/api/process")
    async def process(
        image: UploadFile = File(...),
        sizeId: str = Form(...),
        backgroundColor: str = Form(...),
        faceX: float | None = Form(default=None),
        faceY: float | None = Form(default=None),
        faceW: float | None = Form(default=None),
        faceH: float | None = Form(default=None),
        cropX: float | None = Form(default=None),
        cropY: float | None = Form(default=None),
        cropW: float | None = Form(default=None),
        cropH: float | None = Form(default=None),
    ) -> JSONResponse:
        data = await image.read()

        mime = image.content_type or ""
        upload_error = _validate_upload(data, mime)
        if upload_error is not None:
            return upload_error

        # Validate sizeId
        selected_size = SIZE_OPTIONS_BY_ID.get(sizeId)
        if not selected_size:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": "Invalid sizeId. See /api/options for valid values.",
                        }
                    ],
                },
            )

        # Validate backgroundColor
        if not backgroundColor or not re.fullmatch(
            r"#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})", backgroundColor
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": (
                                "backgroundColor must be a valid hex colour (e.g. #FF0000)."
                            ),
                        }
                    ],
                },
            )

        # Validate that face fields are either all present or all absent.
        # Partial fields (e.g. due to a client bug) are rejected rather than
        # silently ignored, which would produce an uncropped output.
        face_fields = {"faceX": faceX, "faceY": faceY, "faceW": faceW, "faceH": faceH}
        provided = [k for k, v in face_fields.items() if v is not None]
        if 0 < len(provided) < 4:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": f"Partial face fields supplied ({', '.join(provided)}). "
                            "Either supply all four (faceX, faceY, faceW, faceH) or none.",
                        }
                    ],
                },
            )

        normalised_face: NormalisedFace | None = None
        if len(provided) == 4:
            # Validate that all coordinates are within the normalised 0–1 range.
            if not (0.0 <= faceX <= 1.0 and 0.0 <= faceY <= 1.0):  # type: ignore[operator]
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "errors": [
                            {
                                "type": "validation",
                                "message": "faceX and faceY must be in the range [0, 1].",
                            }
                        ],
                    },
                )
            if not (0.0 < faceW <= 1.0 and 0.0 < faceH <= 1.0):  # type: ignore[operator]
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "errors": [
                            {
                                "type": "validation",
                                "message": "faceW and faceH must be in the range (0, 1].",
                            }
                        ],
                    },
                )
            normalised_face = NormalisedFace(
                x=faceX,  # type: ignore[arg-type]
                y=faceY,  # type: ignore[arg-type]
                width=faceW,  # type: ignore[arg-type]
                height=faceH,  # type: ignore[arg-type]
            )

        # Validate crop area fields (all or none; takes priority over face fields).
        crop_fields = {"cropX": cropX, "cropY": cropY, "cropW": cropW, "cropH": cropH}
        provided_crop = [k for k, v in crop_fields.items() if v is not None]
        if 0 < len(provided_crop) < 4:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": (
                                f"Partial crop fields supplied ({', '.join(provided_crop)}). "
                                "Either supply all four (cropX, cropY, cropW, cropH) or none."
                            ),
                        }
                    ],
                },
            )

        normalised_crop_area: NormalisedCropArea | None = None
        if len(provided_crop) == 4:
            if not (0.0 <= cropX <= 1.0 and 0.0 <= cropY <= 1.0):  # type: ignore[operator]
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "errors": [
                            {
                                "type": "validation",
                                "message": "cropX and cropY must be in the range [0, 1].",
                            }
                        ],
                    },
                )
            if not (0.0 < cropW <= 1.0 and 0.0 < cropH <= 1.0):  # type: ignore[operator]
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "errors": [
                            {
                                "type": "validation",
                                "message": "cropW and cropH must be in the range (0, 1].",
                            }
                        ],
                    },
                )
            normalised_crop_area = NormalisedCropArea(
                x=cropX,  # type: ignore[arg-type]
                y=cropY,  # type: ignore[arg-type]
                width=cropW,  # type: ignore[arg-type]
                height=cropH,  # type: ignore[arg-type]
            )

        orch_result = process_image(
            image_data=data,
            mime_type=mime,
            selected_size=selected_size,
            background_color=backgroundColor,
            u2net_model=models.u2net,
            normalised_face=normalised_face,
            normalised_crop_area=normalised_crop_area,
        )

        if orch_result.errors:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "errors": [
                        {"type": e.type, "message": e.message}
                        for e in orch_result.errors
                    ],
                },
            )

        r = orch_result.result
        if r is None:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "processing",
                            "message": "An unexpected error occurred.",
                        }
                    ],
                },
            )
        return JSONResponse(
            content={
                "success": True,
                "warnings": orch_result.warnings,
                "idPhoto": r.id_photo_b64,
            }
        )

    # ── POST /api/layout ──────────────────────────────────────────────────────

    @app.post("/api/layout")
    async def layout(
        image: UploadFile = File(...),
        sizeId: str = Form(...),
        paperType: str = Form(...),
    ) -> JSONResponse:
        data = await image.read()

        mime = image.content_type or ""
        upload_error = _validate_upload(data, mime)
        if upload_error is not None:
            return upload_error

        selected_size = SIZE_OPTIONS_BY_ID.get(sizeId)
        if not selected_size:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": "Invalid sizeId. See /api/options for valid values.",
                        }
                    ],
                },
            )

        if paperType not in ALLOWED_PAPER_TYPES:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": 'paperType must be "6-inch" or "a4".',
                        }
                    ],
                },
            )

        try:
            layout_data = generate_print_layout(
                photo_data=data,
                photo_size=PhotoSize(
                    width_mm=selected_size.physical_width,
                    height_mm=selected_size.physical_height,
                ),
                paper_type=paperType,
                dpi=config.REQUIRED_DPI,
            )
        except Exception:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "processing",
                            "message": "Failed to generate print layout.",
                        }
                    ],
                },
            )

        return JSONResponse(
            content={
                "success": True,
                "printLayout": base64.b64encode(layout_data).decode(),
            }
        )

    return app
