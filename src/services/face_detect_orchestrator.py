from dataclasses import dataclass, field

from src.services.face_detection_service import FaceDetectionModel, detect_faces_in_buffer
from src.services.image_scaling import scale_image_to_target
from src.services.image_validation import validate_image_buffer
from src.utils.crop_area_calculation import FaceBox


@dataclass
class DetectFaceError:
    type: str  # "validation" | "face-detection"
    message: str


@dataclass
class DetectFaceResult:
    image_width: int | None = None
    image_height: int | None = None
    face: FaceBox | None = None
    errors: list[DetectFaceError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_face_in_image(
    image_data: bytes,
    mime_type: str,
    face_detection_model: FaceDetectionModel,
) -> DetectFaceResult:
    """Validate image, run face detection, and return the face bounding box + image dimensions."""
    warnings: list[str] = []

    # Step 1: Validate
    validation = validate_image_buffer(image_data, mime_type)
    if not validation.is_valid:
        return DetectFaceResult(
            errors=[DetectFaceError(type="validation", message=m) for m in validation.errors]
        )
    warnings.extend(validation.warnings)

    # Step 2: Scale down if needed
    buf = scale_image_to_target(image_data) if validation.needs_scaling else image_data

    # Step 3: Face detection
    face_result = detect_faces_in_buffer(face_detection_model, buf)

    if len(face_result.faces) == 0:
        return DetectFaceResult(
            errors=[DetectFaceError(
                type="face-detection",
                message="No face detected. Please upload an image with exactly one face.",
            )],
            warnings=warnings,
        )
    if len(face_result.faces) > 1:
        return DetectFaceResult(
            errors=[DetectFaceError(
                type="face-detection",
                message="Multiple faces detected. Please upload an image with exactly one face.",
            )],
            warnings=warnings,
        )

    from io import BytesIO

    from PIL import Image
    img = Image.open(BytesIO(buf))
    w, h = img.size

    return DetectFaceResult(
        image_width=w,
        image_height=h,
        face=face_result.faces[0],
        warnings=warnings,
    )
