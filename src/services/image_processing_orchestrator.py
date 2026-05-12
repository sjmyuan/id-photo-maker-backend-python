import base64
import sys
import traceback
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

from src import config
from src.services.exact_crop_service import generate_exact_crop_from_image
from src.services.image_validation import validate_image_buffer
from src.services.u2net_service import U2NetModel, remove_background
from src.types.index import SizeOption
from src.utils.crop_area_calculation import (
    CropArea,
    FaceBox,
    calculate_initial_crop_area,
)
from src.utils.dpi_calculation import calculate_dpi


@dataclass
class NormalisedFace:
    """Face bounding box with coordinates normalised to 0.0–1.0 relative to the image."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class NormalisedCropArea:
    """Pre-computed crop area with coordinates normalised to 0.0–1.0 relative to the image."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class ProcessingError:
    type: str  # "validation" | "matting" | "processing"
    message: str


@dataclass
class ProcessingResult:
    id_photo_b64: str


@dataclass
class OrchestratorResult:
    result: ProcessingResult | None = None
    errors: list[ProcessingError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    clean = hex_color.lstrip("#")
    if len(clean) == 3:
        clean = clean[0] * 2 + clean[1] * 2 + clean[2] * 2
    return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


def process_image(
    image_data: bytes,
    mime_type: str,
    selected_size: SizeOption,
    background_color: str,
    u2net_model: U2NetModel,
    required_dpi: int = config.REQUIRED_DPI,
    normalised_face: NormalisedFace | None = None,
    normalised_crop_area: NormalisedCropArea | None = None,
) -> OrchestratorResult:
    """Process an image: (optional crop) → matting → exact resize → background.

    Crop priority (highest to lowest):
    1. ``normalised_crop_area``: pre-computed crop from /api/detect — used directly,
       skipping ``calculate_initial_crop_area``.
    2. ``normalised_face``: face bbox; crop area is computed server-side.
    3. No crop info: image is used as-is.
    """
    warnings: list[str] = []

    try:
        # Step 1: Validate MIME type and readability
        validation = validate_image_buffer(image_data, mime_type)
        if not validation.is_valid:
            return OrchestratorResult(
                errors=[
                    ProcessingError(type="validation", message=m)
                    for m in validation.errors
                ]
            )

        # Step 2: Decode the image and apply the initial face crop.
        img_w, img_h = validation.dimensions
        source_img = Image.open(BytesIO(image_data))

        if normalised_crop_area is not None:
            # Fast path: pre-computed crop area — convert to pixels and crop directly.
            left = max(0, round(normalised_crop_area.x * img_w))
            top = max(0, round(normalised_crop_area.y * img_h))
            right = min(
                img_w,
                round((normalised_crop_area.x + normalised_crop_area.width) * img_w),
            )
            bottom = min(
                img_h,
                round((normalised_crop_area.y + normalised_crop_area.height) * img_h),
            )
            source_img = source_img.crop((left, top, right, bottom))
            img_w, img_h = source_img.size

        elif normalised_face is not None:
            # Legacy path: compute crop area from face bbox server-side.
            face = FaceBox(
                x=normalised_face.x * img_w,
                y=normalised_face.y * img_h,
                width=normalised_face.width * img_w,
                height=normalised_face.height * img_h,
            )
            crop_area = calculate_initial_crop_area(
                face, selected_size.aspect_ratio, img_w, img_h
            )
            left = max(0, round(crop_area.x))
            top = max(0, round(crop_area.y))
            right = min(img_w, round(crop_area.x + crop_area.width))
            bottom = min(img_h, round(crop_area.y + crop_area.height))
            source_img = source_img.crop((left, top, right, bottom))
            img_w, img_h = source_img.size

        # Step 3: Defence-in-depth DPI check on the actual crop dimensions.
        # Warn if the output will be below the required DPI, but do not block
        # processing (the frontend already validated this).
        dpi_result = calculate_dpi(
            img_w,
            img_h,
            selected_size.physical_width,
            selected_size.physical_height,
        )
        if dpi_result.min_dpi < required_dpi:
            calc_dpi = round(dpi_result.min_dpi)
            warnings.append(
                f"Image resolution ({calc_dpi} DPI) is below the recommended "
                f"{required_dpi} DPI. Output quality may be reduced."
            )

        # Step 4: Background removal via rembg/U2Net.
        # Pass a PIL Image in and receive one back to avoid rembg's internal
        # PNG decode and re-encode (saves ~2 full image decode/encode cycles).
        trans_img = remove_background(source_img, u2net_model)

        # Step 5: Exact crop to final pixel dimensions.
        exact_img = generate_exact_crop_from_image(
            trans_img,
            CropArea(x=0, y=0, width=img_w, height=img_h),
            selected_size.physical_width,
            selected_size.physical_height,
            required_dpi,
        )

        # Step 6: Apply background colour (exact_img is already a PIL Image — no re-decode).
        exact_rgba = exact_img.convert("RGBA")
        photo_w, photo_h = exact_rgba.size
        bg = Image.new(
            "RGBA", (photo_w, photo_h), (*_hex_to_rgb(background_color), 255)
        )
        composite = Image.alpha_composite(bg, exact_rgba).convert("RGB")
        id_photo_buf = BytesIO()
        composite.save(
            id_photo_buf, format="JPEG", quality=95, dpi=(required_dpi, required_dpi)
        )
        id_photo_data = id_photo_buf.getvalue()

        return OrchestratorResult(
            result=ProcessingResult(
                id_photo_b64=base64.b64encode(id_photo_data).decode(),
            ),
            warnings=warnings,
        )

    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return OrchestratorResult(
            errors=[
                ProcessingError(
                    type="processing", message="An internal processing error occurred."
                )
            ]
        )
