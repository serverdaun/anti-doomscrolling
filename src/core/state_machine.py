import logging
import time
from enum import Enum

from src.core.config import VisionConfig

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "IDLE"
    SUSPECT = "SUSPECT"
    DOOMSCROLLING = "DOOMSCROLLING"
    CLEARING = "CLEARING"


class DoomscrollStateMachine:
    def __init__(self, config: VisionConfig | None = None):
        self._config = config or VisionConfig()
        self.state = State.IDLE
        self._suspect_start: float | None = None
        self._clearing_start: float | None = None

    def update(
        self, phone_detected: bool, looking_at_screen: bool, face_detected: bool
    ) -> State:
        doomscrolling_signal = phone_detected and (
            not looking_at_screen or not face_detected
        )
        now = time.monotonic()
        prev_state = self.state

        if self.state == State.IDLE:
            if doomscrolling_signal:
                self.state = State.SUSPECT
                self._suspect_start = now

        elif self.state == State.SUSPECT:
            if not doomscrolling_signal:
                self.state = State.IDLE
                self._suspect_start = None
            elif now - self._suspect_start >= self._config.confirm_duration:
                self.state = State.DOOMSCROLLING

        elif self.state == State.DOOMSCROLLING:
            if not doomscrolling_signal:
                self.state = State.CLEARING
                self._clearing_start = now

        elif self.state == State.CLEARING:
            if doomscrolling_signal:
                self.state = State.DOOMSCROLLING
                self._clearing_start = None
            elif now - self._clearing_start >= self._config.clear_duration:
                self.state = State.IDLE
                self._clearing_start = None

        if self.state != prev_state:
            logger.info("State: %s → %s", prev_state.value, self.state.value)

        return self.state
