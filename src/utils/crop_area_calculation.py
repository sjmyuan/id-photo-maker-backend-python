from dataclasses import dataclass


@dataclass
class FaceBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class CropArea:
    x: float
    y: float
    width: float
    height: float


def calculate_initial_crop_area(
    face: FaceBox,
    aspect_ratio: float,
    image_width: int,
    image_height: int,
) -> CropArea:
    """
    Expand around the detected face to include head and shoulders.

    The crop centre is shifted upward relative to the face centre so that
    ``vertical_above`` space appears above the face box (for the crown/hair)
    and ``vertical_below`` space appears below (for the neck/shoulders).
    Shrinks proportionally if it would exceed the image bounds.
    """
    face_center_x = face.x + face.width / 2
    face_center_y = face.y + face.height / 2

    horizontal_expansion = face.width * 0.4
    vertical_above = face.height * 1.0
    vertical_below = face.height * 0.6

    target_w = face.width + 2 * horizontal_expansion
    target_h = face.height + vertical_above + vertical_below

    if (target_w / target_h) > aspect_ratio:
        crop_w = target_w
        crop_h = crop_w / aspect_ratio
    else:
        crop_h = target_h
        crop_w = crop_h * aspect_ratio

    # Shift the crop centre upward so the asymmetric vertical margins are honoured.
    # Without this shift the crop would be centred on the face and both the
    # vertical_above and vertical_below values would contribute equally (their
    # average) to the top and bottom margins, ignoring the intended asymmetry.
    crop_center_y = face_center_y + (vertical_below - vertical_above) / 2

    clamped_cx = max(0.0, min(face_center_x, float(image_width)))
    clamped_cy = max(0.0, min(crop_center_y, float(image_height)))

    crop_x = clamped_cx - crop_w / 2
    crop_y = clamped_cy - crop_h / 2

    exceeds = (
        crop_x < 0
        or crop_x + crop_w > image_width
        or crop_y < 0
        or crop_y + crop_h > image_height
    )

    if exceeds:
        max_w = min(clamped_cx, image_width - clamped_cx) * 2
        max_h = min(clamped_cy, image_height - clamped_cy) * 2

        if max_w <= 0 or max_h <= 0:
            # Degenerate: crop centre is at or beyond the image edge.
            # Fit the largest aspect-ratio-correct crop from the top-left corner.
            crop_w = min(crop_w, float(image_width))
            crop_h = min(crop_h, float(image_height))
            if crop_w / crop_h > aspect_ratio:
                crop_w = crop_h * aspect_ratio
            else:
                crop_h = crop_w / aspect_ratio
            return CropArea(x=0.0, y=0.0, width=crop_w, height=crop_h)

        if max_w / aspect_ratio <= max_h:
            crop_w = max_w
            crop_h = max_w / aspect_ratio
        else:
            crop_h = max_h
            crop_w = max_h * aspect_ratio

        crop_x = clamped_cx - crop_w / 2
        crop_y = clamped_cy - crop_h / 2

    return CropArea(x=crop_x, y=crop_y, width=crop_w, height=crop_h)
