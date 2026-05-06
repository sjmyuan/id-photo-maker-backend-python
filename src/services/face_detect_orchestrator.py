from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

from src.services.face_detection_service import (
    FaceDetectionModel,
    detect_faces_in_buffer,
)
from src.services.image_scaling import scale_image_to_target
from src.services.image_validation import validate_image_buffer


@dataclass
class DetectFaceError:
    type: str  # "validation" | "face-detection"
    message: str


@dataclass
class NormalisedFaceBox:
    """Face bounding box with coordinates normalised to 0.0–1.0 relative to the image."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class DetectFaceResult:
    face: NormalisedFaceBox | None = None
    errors: list[DetectFaceError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_face_in_image(
    image_data: bytes,
    mime_type: str,
    face_detection_model: FaceDetectionModel,
) -> DetectFaceResult:
    """Validate image, run face detection, and return a normalised face bounding box (0–1)."""
    warnings: list[str] = []

    # Step 1: Validate
    validation = validate_image_buffer(image_data, mime_type)
    if not validation.is_valid:
        return DetectFaceResult(
            errors=[
                DetectFaceError(type="validation", message=m) for m in validation.errors
            ]
        )
    warnings.extend(validation.warnings)

    # Step 2: Scale down if needed (face detection does not require full resolution)
    buf = scale_image_to_target(image_data) if validation.needs_scaling else image_data

    # Step 3: Determine image dimensions for normalisation
    img = Image.open(BytesIO(buf))
    img_w, img_h = img.size

    # Step 4: Face detection
    face_result = detect_faces_in_buffer(face_detection_model, buf)

    if len(face_result.faces) == 0:
        return DetectFaceResult(
            errors=[
                DetectFaceError(
                    type="face-detection",
                    message="No face detected. Please upload an image with exactly one face.",
                )
            ],
            warnings=warnings,
        )
    if len(face_result.faces) > 1:
        return DetectFaceResult(
            errors=[
                DetectFaceError(
                    type="face-detection",
                    message=(
                        "Multiple faces detected. Please upload an image with exactly one face."
                    ),
                )
            ],
            warnings=warnings,
        )

    # Step 5: Normalise absolute pixel bbox → 0–1 range
    face_px = face_result.faces[0]
    normalised = NormalisedFaceBox(
        x=face_px.x / img_w,
        y=face_px.y / img_h,
        width=face_px.width / img_w,
        height=face_px.height / img_h,
    )

    return DetectFaceResult(
        face=normalised,
        warnings=warnings,
    )
