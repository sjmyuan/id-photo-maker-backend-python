import base64
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

from src.services.exact_crop_service import generate_exact_crop
from src.services.face_detection_service import (
    FaceDetectionModel,
    detect_faces_in_buffer,
)
from src.services.image_scaling import scale_image_to_target
from src.services.image_validation import validate_image_buffer
from src.services.print_layout_service import generate_print_layout
from src.services.u2net_service import U2NetModel, remove_background
from src.types.index import PaperMargins, SizeOption
from src.utils.crop_area_calculation import CropArea, calculate_initial_crop_area
from src.utils.dpi_calculation import calculate_dpi
from src.utils.layout_calculation import PhotoSize


@dataclass
class ProcessingError:
    type: str  # "validation" | "face-detection" | "dpi" | "matting" | "processing"
    message: str


@dataclass
class ProcessingResult:
    id_photo_b64: str
    print_layout_b64: str


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
    paper_type: str,
    margins: PaperMargins,
    u2net_model: U2NetModel,
    face_detection_model: FaceDetectionModel,
    required_dpi: int = 300,
) -> OrchestratorResult:
    warnings: list[str] = []

    try:
        # Step 1: Validate
        validation = validate_image_buffer(image_data, mime_type)
        if not validation.is_valid:
            return OrchestratorResult(
                errors=[
                    ProcessingError(type="validation", message=m)
                    for m in validation.errors
                ]
            )
        warnings.extend(validation.warnings)

        # Step 2: Scale down if needed
        buf = (
            scale_image_to_target(image_data)
            if validation.needs_scaling
            else image_data
        )

        # Step 3: Face detection
        face_result = detect_faces_in_buffer(face_detection_model, buf)

        if len(face_result.faces) == 0:
            return OrchestratorResult(
                errors=[
                    ProcessingError(
                        type="face-detection",
                        message="No face detected. Please upload an image with exactly one face.",
                    )
                ],
                warnings=warnings,
            )
        if len(face_result.faces) > 1:
            return OrchestratorResult(
                errors=[
                    ProcessingError(
                        type="face-detection",
                        message=(
                            "Multiple faces detected. Please upload an image with exactly one face."
                        ),
                    )
                ],
                warnings=warnings,
            )

        # Step 4: Calculate crop area
        face = face_result.faces[0]
        img = Image.open(BytesIO(buf))
        img_w, img_h = img.size

        crop_area: CropArea = calculate_initial_crop_area(
            face, selected_size.aspect_ratio, img_w, img_h
        )

        # Step 5: Validate DPI
        dpi_result = calculate_dpi(
            crop_area.width,
            crop_area.height,
            selected_size.physical_width,
            selected_size.physical_height,
        )
        if dpi_result.min_dpi < required_dpi:
            calc_dpi = round(dpi_result.min_dpi)
            return OrchestratorResult(
                errors=[
                    ProcessingError(
                        type="dpi",
                        message=(
                            f"DPI requirement ({required_dpi} DPI) cannot be met. "
                            f"Calculated DPI: {calc_dpi}. "
                            "Please upload a higher-resolution image."
                        ),
                    )
                ],
                warnings=warnings,
            )

        # Step 6: Crop to face area
        left = max(0, round(crop_area.x))
        top = max(0, round(crop_area.y))
        right = left + round(crop_area.width)
        bottom = top + round(crop_area.height)
        cropped_img = img.crop((left, top, right, bottom))
        cropped_buf = BytesIO()
        cropped_img.save(cropped_buf, format="PNG")
        cropped_data = cropped_buf.getvalue()

        # Step 7: Background removal via rembg/U2Net
        transparent_data = remove_background(cropped_data, u2net_model)

        # Step 8: Exact crop to final pixel dimensions
        trans_img = Image.open(BytesIO(transparent_data))
        trans_w, trans_h = trans_img.size
        exact_data = generate_exact_crop(
            transparent_data,
            CropArea(x=0, y=0, width=trans_w, height=trans_h),
            selected_size.physical_width,
            selected_size.physical_height,
            required_dpi,
        )

        # Step 9: Apply background colour
        exact_img = Image.open(BytesIO(exact_data)).convert("RGBA")
        photo_w, photo_h = exact_img.size
        bg = Image.new(
            "RGBA", (photo_w, photo_h), (*_hex_to_rgb(background_color), 255)
        )
        composite = Image.alpha_composite(bg, exact_img).convert("RGB")
        id_photo_buf = BytesIO()
        composite.save(id_photo_buf, format="PNG")
        id_photo_data = id_photo_buf.getvalue()

        # Step 10: Print layout
        print_layout_data = generate_print_layout(
            id_photo_data,
            PhotoSize(
                width_mm=selected_size.physical_width,
                height_mm=selected_size.physical_height,
            ),
            paper_type,
            required_dpi,
            margins,
        )

        return OrchestratorResult(
            result=ProcessingResult(
                id_photo_b64=base64.b64encode(id_photo_data).decode(),
                print_layout_b64=base64.b64encode(print_layout_data).decode(),
            ),
            warnings=warnings,
        )

    except Exception as e:
        return OrchestratorResult(
            errors=[ProcessingError(type="processing", message=str(e))],
            warnings=warnings,
        )
