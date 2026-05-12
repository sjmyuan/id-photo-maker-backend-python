from dataclasses import dataclass, field

from src import config
from src.services.face_detection_service import (
    FaceDetectionModel,
    detect_faces_in_buffer,
)
from src.services.image_scaling import scale_image_to_target
from src.services.image_validation import validate_image_buffer
from src.types.index import SizeOption
from src.utils.crop_area_calculation import (
    CONSTRAINTS_BY_SIZE,
    FaceBox,
    calculate_initial_crop_area,
)
from src.utils.dpi_calculation import calculate_dpi


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
class NormalisedCropArea:
    """Crop area with coordinates normalised to 0.0–1.0 relative to the image."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class DPICheck:
    """Result of a DPI sufficiency check for the computed crop area."""

    dpi: float
    sufficient: bool


@dataclass
class DetectFaceResult:
    face: NormalisedFaceBox | None = None
    crop_area: NormalisedCropArea | None = None
    dpi_check: DPICheck | None = None
    errors: list[DetectFaceError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_face_in_image(
    image_data: bytes,
    mime_type: str,
    face_detection_model: FaceDetectionModel,
    *,
    size_option: SizeOption | None = None,
) -> DetectFaceResult:
    """Validate image, run face detection, and return a normalised face bounding box (0–1).

    When ``size_option`` is provided the function additionally computes the
    crop area (normalised 0–1) for that photo size and performs a DPI check.
    """
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

    # Step 2: Scale down if needed (face detection does not require full resolution).
    # When no scaling is required the image dimensions are already known from
    # validation, avoiding a redundant Image.open call just to read the size.
    if validation.needs_scaling:
        buf, (img_w, img_h) = scale_image_to_target(image_data)
    else:
        buf = image_data
        img_w, img_h = validation.dimensions

    # Step 3: Face detection
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

    # Step 4: Normalise absolute pixel bbox → 0–1 range
    face_px = face_result.faces[0]
    # validation.dimensions holds the original (pre-scaling) image size — use it
    # directly instead of re-opening the original buffer.
    orig_w, orig_h = validation.dimensions

    # The face detector ran on the scaled buffer; scale the bbox back to
    # original image space before normalising.
    scale_x = orig_w / img_w
    scale_y = orig_h / img_h
    normalised = NormalisedFaceBox(
        x=(face_px.x * scale_x) / orig_w,
        y=(face_px.y * scale_y) / orig_h,
        width=(face_px.width * scale_x) / orig_w,
        height=(face_px.height * scale_y) / orig_h,
    )

    # Step 5 (optional): Compute crop area + DPI check when a size is requested.
    crop_area: NormalisedCropArea | None = None
    dpi_check: DPICheck | None = None

    if size_option is not None:
        face_abs = FaceBox(
            x=normalised.x * orig_w,
            y=normalised.y * orig_h,
            width=normalised.width * orig_w,
            height=normalised.height * orig_h,
        )
        ca_px = calculate_initial_crop_area(
            face_abs,
            size_option.aspect_ratio,
            orig_w,
            orig_h,
            constraints=CONSTRAINTS_BY_SIZE.get(size_option.id),
        )
        crop_area = NormalisedCropArea(
            x=ca_px.x / orig_w,
            y=ca_px.y / orig_h,
            width=ca_px.width / orig_w,
            height=ca_px.height / orig_h,
        )

        dpi_result = calculate_dpi(
            ca_px.width,
            ca_px.height,
            size_option.physical_width,
            size_option.physical_height,
        )
        dpi_check = DPICheck(
            dpi=dpi_result.min_dpi, sufficient=dpi_result.min_dpi >= config.REQUIRED_DPI
        )

    return DetectFaceResult(
        face=normalised,
        crop_area=crop_area,
        dpi_check=dpi_check,
        warnings=warnings,
    )
