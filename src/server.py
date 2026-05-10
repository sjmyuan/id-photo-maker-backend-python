import re
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import config
from src.services.face_detect_orchestrator import detect_face_in_image
from src.services.face_detection_service import FaceDetectionModel
from src.services.image_processing_orchestrator import NormalisedFace, process_image
from src.services.print_layout_service import generate_print_layout
from src.services.u2net_service import U2NetModel
from src.types.index import SIZE_OPTIONS_BY_ID
from src.utils.layout_calculation import PhotoSize

ALLOWED_PAPER_TYPES = {"6-inch", "a4"}

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


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

    # ── POST /api/detect ──────────────────────────────────────────────────────

    @app.post("/api/detect")
    async def detect(image: UploadFile = File(...)) -> JSONResponse:
        data = await image.read()

        max_mb = config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        if len(data) > config.MAX_UPLOAD_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": (
                                f"File too large. Maximum allowed size is {max_mb} MB."
                            ),
                        }
                    ],
                },
            )

        mime = image.content_type or ""
        if mime not in ALLOWED_MIME_TYPES:
            return JSONResponse(
                status_code=415,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": (
                                "Invalid file type. Only JPEG, PNG, and WebP are supported."
                            ),
                        }
                    ],
                },
            )

        result = detect_face_in_image(data, mime, models.face_detection)

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

        face = result.face
        face_dict = (
            {"x": face.x, "y": face.y, "width": face.width, "height": face.height}
            if face
            else None
        )
        return JSONResponse(
            content={
                "success": True,
                "face": face_dict,
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
    ) -> JSONResponse:
        data = await image.read()

        if len(data) > config.MAX_UPLOAD_SIZE_BYTES:
            max_mb = config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": (
                                f"File too large. Maximum allowed size is {max_mb} MB."
                            ),
                        }
                    ],
                },
            )

        mime = image.content_type or ""
        if mime not in ALLOWED_MIME_TYPES:
            return JSONResponse(
                status_code=415,
                content={
                    "success": False,
                    "errors": [
                        {
                            "type": "validation",
                            "message": (
                                "Invalid file type. Only JPEG, PNG, and WebP are supported."
                            ),
                        }
                    ],
                },
            )

        # Validate sizeId
        selected_size = SIZE_OPTIONS_BY_ID.get(sizeId)
        if not selected_size:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "errors": [
                        {"type": "validation", "message": f"Invalid sizeId: {sizeId}"}
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

        normalised_face: NormalisedFace | None = None
        if (
            faceX is not None
            and faceY is not None
            and faceW is not None
            and faceH is not None
        ):
            normalised_face = NormalisedFace(
                x=faceX, y=faceY, width=faceW, height=faceH
            )

        orch_result = process_image(
            image_data=data,
            mime_type=mime,
            selected_size=selected_size,
            background_color=backgroundColor,
            u2net_model=models.u2net,
            normalised_face=normalised_face,
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
        assert r is not None
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

        max_mb = config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        if len(data) > config.MAX_UPLOAD_SIZE_BYTES:
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

        mime = image.content_type or ""
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

        selected_size = SIZE_OPTIONS_BY_ID.get(sizeId)
        if not selected_size:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "errors": [
                        {"type": "validation", "message": f"Invalid sizeId: {sizeId}"}
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
                dpi=300,
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

        import base64

        return JSONResponse(
            content={
                "success": True,
                "printLayout": base64.b64encode(layout_data).decode(),
            }
        )

    return app
