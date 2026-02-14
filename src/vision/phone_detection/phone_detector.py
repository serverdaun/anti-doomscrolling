import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from src.core.config import VisionConfig

logger = logging.getLogger(__name__)


@dataclass
class PhoneDetection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class PhoneDetector:
    def __init__(self, config: VisionConfig | None = None):
        self._config = config or VisionConfig()
        try:
            self._model = YOLO(self._config.yolo_model)
            self._model.to(self._config.yolo_device)
            logger.info("YOLO loaded on %s", self._config.yolo_device)
        except Exception:
            logger.warning("Failed to load YOLO on %s, falling back to CPU",
                           self._config.yolo_device, exc_info=True)
            self._model = YOLO(self._config.yolo_model)
            logger.info("YOLO loaded on CPU (fallback)")

    def detect(self, frame: np.ndarray) -> list[PhoneDetection]:
        results = self._model(
            frame,
            conf=self._config.yolo_confidence,
            classes=[self._config.yolo_phone_class],
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                detections.append(
                    PhoneDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                    )
                )
                logger.info("Phone detected: conf=%.2f bbox=(%d,%d,%d,%d)",
                            conf, x1, y1, x2, y2)
        return detections
