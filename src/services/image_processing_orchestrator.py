import base64
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

from src.services.exact_crop_service import generate_exact_crop
from src.services.image_validation import validate_image_buffer
from src.services.u2net_service import U2NetModel, remove_background
from src.types.index import SizeOption
from src.utils.crop_area_calculation import CropArea
from src.utils.dpi_calculation import calculate_dpi


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
    required_dpi: int = 300,
) -> OrchestratorResult:
    """Process a pre-cropped image: matting → exact resize → background → print layout.

    The caller is responsible for sending an already-cropped face region.
    Face detection and crop-area calculation are no longer performed here.
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

        # Step 2: Defence-in-depth DPI check — warn if the crop is too low-res,
        # but do not block processing (the frontend already validated this).
        img_w, img_h = validation.dimensions
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

        # Step 3: Background removal via rembg/U2Net
        transparent_data = remove_background(image_data, u2net_model)

        # Step 4: Exact crop to final pixel dimensions
        trans_img = Image.open(BytesIO(transparent_data))
        trans_w, trans_h = trans_img.size
        exact_data = generate_exact_crop(
            transparent_data,
            CropArea(x=0, y=0, width=trans_w, height=trans_h),
            selected_size.physical_width,
            selected_size.physical_height,
            required_dpi,
        )

        # Step 5: Apply background colour
        exact_img = Image.open(BytesIO(exact_data)).convert("RGBA")
        photo_w, photo_h = exact_img.size
        bg = Image.new(
            "RGBA", (photo_w, photo_h), (*_hex_to_rgb(background_color), 255)
        )
        composite = Image.alpha_composite(bg, exact_img).convert("RGB")
        id_photo_buf = BytesIO()
        composite.save(id_photo_buf, format="PNG")
        id_photo_data = id_photo_buf.getvalue()

        return OrchestratorResult(
            result=ProcessingResult(
                id_photo_b64=base64.b64encode(id_photo_data).decode(),
            ),
            warnings=warnings,
        )

    except Exception as e:
        return OrchestratorResult(
            errors=[ProcessingError(type="processing", message=str(e))],
            warnings=warnings,
        )
