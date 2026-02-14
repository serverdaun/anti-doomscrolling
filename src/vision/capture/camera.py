import logging
import threading

import cv2
import numpy as np

from src.core.config import VisionConfig

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, config: VisionConfig | None = None):
        self._config = config or VisionConfig()
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._config.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._config.camera_index}")
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera %d opened: requested %dx%d, actual %dx%d",
            self._config.camera_index,
            self._config.frame_width,
            self._config.frame_height,
            actual_w,
            actual_h,
        )
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        logger.info("Camera stopped")
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
