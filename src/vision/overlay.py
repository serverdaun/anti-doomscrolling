import cv2
import numpy as np

from src.core.state_machine import State
from src.vision.face_detection.face_detector import FaceResult
from src.vision.gaze_tracking.gaze_estimator import GazeResult
from src.vision.phone_detection.phone_detector import PhoneDetection

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_CYAN = (255, 255, 0)
_WHITE = (255, 255, 255)
_YELLOW = (0, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Iris landmark indices
_LEFT_IRIS = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]
# Eye contour landmarks — subset
_LEFT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
_RIGHT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]


class Overlay:
    def draw(
        self,
        frame: np.ndarray,
        face: FaceResult | None,
        gaze: GazeResult | None,
        phones: list[PhoneDetection],
        state: State,
        fps: float,
        process_ms: float,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw iris and eye landmarks
        if face is not None:
            pts = face.landmarks_px.astype(int)
            for idx in _LEFT_EYE + _RIGHT_EYE:
                cv2.circle(out, tuple(pts[idx]), 1, _GREEN, -1)
            for idx in _LEFT_IRIS + _RIGHT_IRIS:
                cv2.circle(out, tuple(pts[idx]), 2, _CYAN, -1)

        # Gaze metrics
        if gaze is not None:
            color = _GREEN if gaze.looking_at_screen else _RED
            lines = [
                f"Iris ratio: {gaze.iris_ratio_avg:.2f}",
                f"Head pitch: {gaze.head_pitch:.1f} deg",
                f"Looking at screen: {gaze.looking_at_screen}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(out, line, (10, 30 + i * 25), _FONT, 0.6, color, 2)
        elif face is None:
            cv2.putText(out, "No face detected", (10, 30), _FONT, 0.6, _YELLOW, 2)

        # Phone bounding boxes
        for phone in phones:
            x1, y1, x2, y2 = phone.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), _RED, 2)
            label = f"phone {phone.confidence:.0%}"
            cv2.putText(out, label, (x1, y1 - 8), _FONT, 0.5, _RED, 2)

        # State indicator
        state_text = {
            State.IDLE: "OK",
            State.SUSPECT: "DETECTING...",
            State.DOOMSCROLLING: "DOOMSCROLLING",
            State.CLEARING: "CLEARING...",
        }[state]
        state_color = _RED if state in (State.DOOMSCROLLING, State.SUSPECT) else _GREEN
        text_size = cv2.getTextSize(state_text, _FONT, 1.0, 2)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(out, state_text, (tx, h - 20), _FONT, 1.0, state_color, 2)

        # Red border when doomscrolling
        if state == State.DOOMSCROLLING:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), _RED, 4)

        # FPS counter
        fps_text = f"{fps:.1f} FPS ({process_ms:.0f}ms)"
        cv2.putText(out, fps_text, (w - 220, 30), _FONT, 0.6, _WHITE, 2)

        return out
