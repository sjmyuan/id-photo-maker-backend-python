from dataclasses import dataclass
from io import BytesIO

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

from src import config
from src.utils.crop_area_calculation import FaceBox

FACE_DETECTOR_MODEL_FILENAME = "blaze_face_short_range.tflite"


@dataclass
class FaceDetectionModel:
    detector: object
    status: str = "loaded"


@dataclass
class FaceDetectionResult:
    faces: list[FaceBox]
    error: str | None = None


def load_face_detection_model() -> FaceDetectionModel:
    """Load MediaPipe short-range face detection model via Tasks API."""
    model_path = str(config.MODELS_DIR / FACE_DETECTOR_MODEL_FILENAME)
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
    )
    detector = mp_vision.FaceDetector.create_from_options(options)
    return FaceDetectionModel(detector=detector)


def detect_faces_in_buffer(
    model: FaceDetectionModel, image_data: bytes
) -> FaceDetectionResult:
    """Detect faces in an image buffer and return bounding boxes."""
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img_array = np.array(img)
    width, height = img.size

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    results = model.detector.detect(mp_image)  # type: ignore[attr-defined]

    if not results.detections:
        return FaceDetectionResult(faces=[], error="no-face-detected")

    faces: list[FaceBox] = []
    for detection in results.detections:
        # Tasks API returns absolute pixel coordinates
        bbox = detection.bounding_box
        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)
        w = min(bbox.width, width - x)
        h = min(bbox.height, height - y)
        faces.append(FaceBox(x=x, y=y, width=w, height=h))

    error: str | None = None
    if len(faces) > 1:
        error = "multiple-faces-detected"

    return FaceDetectionResult(faces=faces, error=error)
