import logging
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from src.core.config import VisionConfig

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass
class FaceResult:
    landmarks_px: np.ndarray  # (478, 2) pixel coords
    landmarks_norm: np.ndarray  # (478, 3) normalized coords


class FaceDetector:
    def __init__(self, config: VisionConfig | None = None):
        cfg = config or VisionConfig()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.face_model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._first_detection_logged = False
        logger.info("FaceLandmarker loaded from %s", cfg.face_model_path)

    def detect(self, frame: np.ndarray) -> FaceResult | None:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        norm = np.array([(lm.x, lm.y, lm.z) for lm in face])
        px = np.array([(lm.x * w, lm.y * h) for lm in face])
        if not self._first_detection_logged:
            logger.info("First face detected: %d landmarks", len(face))
            self._first_detection_logged = True
        return FaceResult(landmarks_px=px, landmarks_norm=norm)

    def close(self) -> None:
        self._landmarker.close()
