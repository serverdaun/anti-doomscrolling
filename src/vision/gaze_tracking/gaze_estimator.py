import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.core.config import VisionConfig
from src.vision.face_detection.face_detector import FaceResult

logger = logging.getLogger(__name__)

# 3D model points for solvePnP (generic face proportions)
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip (1)
        (0.0, -330.0, -65.0),  # chin (152)
        (-225.0, 170.0, -135.0),  # left eye corner (33)
        (225.0, 170.0, -135.0),  # right eye corner (263)
        (-150.0, -150.0, -125.0),  # left mouth corner (61)
        (150.0, -150.0, -125.0),  # right mouth corner (291)
    ],
    dtype=np.float64,
)

_SOLVEPNP_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Iris landmarks
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473

# Eye corner landmarks
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263


@dataclass
class GazeResult:
    iris_ratio_left: float
    iris_ratio_right: float
    iris_ratio_avg: float
    head_pitch: float
    head_yaw: float
    looking_at_screen: bool


class GazeEstimator:
    def __init__(self, config: VisionConfig | None = None):
        self._config = config or VisionConfig()
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def _get_camera_matrix(self, frame_shape: tuple) -> np.ndarray:
        if self._camera_matrix is None:
            h, w = frame_shape[:2]
            focal_length = w
            center = (w / 2, h / 2)
            self._camera_matrix = np.array(
                [
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        return self._camera_matrix

    def estimate(self, face: FaceResult, frame_shape: tuple) -> GazeResult:
        iris_left = self._iris_ratio(face.landmarks_px, "left")
        iris_right = self._iris_ratio(face.landmarks_px, "right")
        iris_avg = (iris_left + iris_right) / 2.0

        pitch, yaw = self._head_pose(face.landmarks_px, frame_shape)

        horizontal_ok = (
            self._config.iris_ratio_min <= iris_avg <= self._config.iris_ratio_max
        )
        pitch_ok = pitch > self._config.head_pitch_threshold
        looking_at_screen = horizontal_ok and pitch_ok

        logger.debug("iris_avg=%.3f pitch=%.1f looking_at_screen=%s",
                     iris_avg, pitch, looking_at_screen)

        return GazeResult(
            iris_ratio_left=iris_left,
            iris_ratio_right=iris_right,
            iris_ratio_avg=iris_avg,
            head_pitch=pitch,
            head_yaw=yaw,
            looking_at_screen=looking_at_screen,
        )

    def _iris_ratio(self, landmarks_px: np.ndarray, eye: str) -> float:
        if eye == "left":
            iris_center = landmarks_px[_LEFT_IRIS_CENTER]
            inner = landmarks_px[_LEFT_EYE_INNER]
            outer = landmarks_px[_LEFT_EYE_OUTER]
        else:
            iris_center = landmarks_px[_RIGHT_IRIS_CENTER]
            inner = landmarks_px[_RIGHT_EYE_INNER]
            outer = landmarks_px[_RIGHT_EYE_OUTER]

        eye_width = np.linalg.norm(inner - outer)
        if eye_width < 1e-6:
            return 0.5
        iris_dist = np.linalg.norm(iris_center - outer)
        return float(iris_dist / eye_width)

    def _head_pose(
        self, landmarks_px: np.ndarray, frame_shape: tuple
    ) -> tuple[float, float]:
        image_points = np.array(
            [landmarks_px[i] for i in _SOLVEPNP_LANDMARKS],
            dtype=np.float64,
        )
        camera_matrix = self._get_camera_matrix(frame_shape)

        success, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix to Euler angles
        pitch = np.degrees(
            np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
        )
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        return float(pitch), float(yaw)
