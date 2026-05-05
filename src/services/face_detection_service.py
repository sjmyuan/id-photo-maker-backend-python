from dataclasses import dataclass
from io import BytesIO

import mediapipe as mp
import numpy as np
from PIL import Image

from src.utils.crop_area_calculation import FaceBox


@dataclass
class FaceDetectionModel:
    detector: object
    status: str = "loaded"


@dataclass
class FaceDetectionResult:
    faces: list[FaceBox]
    error: str | None = None


def load_face_detection_model() -> FaceDetectionModel:
    """Load MediaPipe short-range face detection model."""
    mp_face = mp.solutions.face_detection  # type: ignore[attr-defined]
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    return FaceDetectionModel(detector=detector)


def detect_faces_in_buffer(
    model: FaceDetectionModel, image_data: bytes
) -> FaceDetectionResult:
    """Detect faces in an image buffer and return bounding boxes."""
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img_array = np.array(img)
    width, height = img.size

    results = model.detector.process(img_array)  # type: ignore[union-attr]

    if not results.detections:
        return FaceDetectionResult(faces=[], error="no-face-detected")

    faces: list[FaceBox] = []
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x = round(bbox.xmin * width)
        y = round(bbox.ymin * height)
        w = round(bbox.width * width)
        h = round(bbox.height * height)
        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        faces.append(FaceBox(x=x, y=y, width=w, height=h))

    error: str | None = None
    if len(faces) > 1:
        error = "multiple-faces-detected"

    return FaceDetectionResult(faces=faces, error=error)
