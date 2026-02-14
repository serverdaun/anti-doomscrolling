import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.core.config import VisionConfig

logger = logging.getLogger(__name__)
from src.vision.face_detection.face_detector import FaceDetector, FaceResult
from src.vision.gaze_tracking.gaze_estimator import GazeEstimator, GazeResult
from src.vision.phone_detection.phone_detector import PhoneDetector, PhoneDetection


@dataclass
class PipelineResult:
    face: FaceResult | None = None
    gaze: GazeResult | None = None
    phones: list[PhoneDetection] = field(default_factory=list)
    process_ms: float = 0.0


class VisionPipeline:
    def __init__(self, config: VisionConfig | None = None):
        self._config = config or VisionConfig()
        self._face_detector = FaceDetector(self._config)
        self._gaze_estimator = GazeEstimator(self._config)
        self._phone_detector = PhoneDetector(self._config)
        self._cycle = 0
        self._last_phones: list[PhoneDetection] = []
        logger.info("VisionPipeline initialized")

    def process(self, frame: np.ndarray) -> PipelineResult:
        t0 = time.perf_counter()

        face = self._face_detector.detect(frame)
        gaze = None
        if face is not None:
            gaze = self._gaze_estimator.estimate(face, frame.shape)

        if self._cycle % self._config.yolo_every_n == 0:
            self._last_phones = self._phone_detector.detect(frame)
        self._cycle += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Pipeline cycle %d: %.1fms", self._cycle, elapsed_ms)
        return PipelineResult(
            face=face,
            gaze=gaze,
            phones=self._last_phones,
            process_ms=elapsed_ms,
        )

    def close(self) -> None:
        self._face_detector.close()
