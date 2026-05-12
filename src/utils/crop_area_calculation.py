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


@dataclass(frozen=True)
class PhotoCropConstraints:
    """Physical positioning constraints for a specific photo size.

    All fractions are expressed relative to the physical photo dimensions
    (width fractions relative to photo width; height fractions relative to
    photo height).  The face bounding box returned by the face detector
    typically starts at the hairline/forehead, not the crown of the head.
    ``_CROWN_FRACTION`` accounts for the hair above the forehead so that the
    final top-margin (crown → upper photo edge) lands on target.
    """

    face_h_target_fraction: float
    """Target face height (head-top to chin) as a fraction of photo height."""

    top_margin_target_fraction: float
    """Target space between head crown and the upper photo boundary, as a
    fraction of photo height."""

    bottom_margin_min_fraction: float
    """Minimum space between chin and the lower photo boundary, as a fraction
    of photo height."""


# Estimated fraction of the face bbox height that the crown/hair extends
# *above* the top of the face bounding box (hairline).  MediaPipe face
# detection typically starts the bbox at the hairline, so we add this
# correction to place the crop top below the crown by the desired margin.
_CROWN_FRACTION: float = 0.18

# ── Per-size constraints ──────────────────────────────────────────────────────
# 33 × 48 mm (large-1-inch, 大一寸)
# • face height (head top → chin): 15 mm – 22 mm  →  target mid = 18.5 mm
# • space above head (crown → upper edge): 3 mm – 5 mm  →  target mid = 4 mm
# • chin → lower edge: > 7 mm
_LARGE_1_INCH_CONSTRAINTS = PhotoCropConstraints(
    face_h_target_fraction=18.5 / 48,
    top_margin_target_fraction=4.0 / 48,
    bottom_margin_min_fraction=7.0 / 48,
)

CONSTRAINTS_BY_SIZE: dict[str, PhotoCropConstraints] = {
    "large-1-inch": _LARGE_1_INCH_CONSTRAINTS,
}


def calculate_initial_crop_area(
    face: FaceBox,
    aspect_ratio: float,
    image_width: int,
    image_height: int,
    *,
    constraints: PhotoCropConstraints | None = None,
) -> CropArea:
    """
    Expand around the detected face to include head and shoulders.

    When ``constraints`` is provided the function uses physical-size-aware
    positioning:

    * The crop height is set so the face occupies ``face_h_target_fraction``
      of the final photo height.
    * The face is shifted upward within the crop so the effective top margin
      (crown → upper boundary) matches ``top_margin_target_fraction``.  A
      ``_CROWN_FRACTION`` correction accounts for the hair above the face bbox.
    * Horizontal expansion is derived from the target crop width (aspect ratio
      applied to the constraint-derived crop height) so the face width scales
      correctly with the rest of the composition.

    When no constraints are provided the function falls back to the original
    relative-multiplier behaviour (vertical_above = 1.0 × face height,
    vertical_below = 0.6 × face height).

    The crop centre is shifted upward relative to the face centre so that
    ``vertical_above`` space appears above the face box (for the crown/hair)
    and ``vertical_below`` space appears below (for the neck/shoulders).
    Shrinks proportionally if it would exceed the image bounds.
    """
    face_center_x = face.x + face.width / 2
    face_center_y = face.y + face.height / 2

    if constraints is not None:
        fhtf = constraints.face_h_target_fraction  # face_height / crop_height
        tmtf = constraints.top_margin_target_fraction  # top_margin / crop_height

        # Space from crop top to face bbox top.  Includes the crown correction
        # so the visible gap between the crown and the photo edge lands on the
        # target top margin.
        vertical_above = face.height * (tmtf / fhtf + _CROWN_FRACTION)

        # Remaining space below the chin.
        vertical_below = max(
            0.0,
            face.height * (1.0 / fhtf - 1.0 - tmtf / fhtf - _CROWN_FRACTION),
        )

        # Derive horizontal expansion from the aspect-ratio-corrected crop
        # width so the height axis stays dominant.
        expected_crop_h = face.height / fhtf
        expected_crop_w = expected_crop_h * aspect_ratio
        horizontal_expansion = max(0.0, (expected_crop_w - face.width) / 2)
    else:
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
