from __future__ import annotations

import logging
import random
import subprocess
import time

from src.core.state_machine import State

logger = logging.getLogger(__name__)

_MESSAGES = [
    "Put the phone down. You were doing something important.",
    "Hey! Phone detected. Eyes up.",
    "Doomscrolling detected. You'll regret this later.",
    "Your future self called — they want their time back.",
    "Phone down. Focus up.",
    "Nothing on that screen is more important than what you're doing.",
]


class MacOSNotifier:
    def __init__(self, cooldown: float = 30.0) -> None:
        self._cooldown = cooldown
        self._last_fired: float = 0.0

    def notify(self, new_state: State, old_state: State) -> None:
        if new_state is not State.DOOMSCROLLING:
            return

        now = time.monotonic()
        if now - self._last_fired < self._cooldown:
            return

        self._last_fired = now
        message = random.choice(_MESSAGES)

        script = (
            f'display notification "{message}" '
            f'with title "Anti-Doomscroll" sound name "Sosumi"'
        )
        try:
            subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Notification sent: %s", message)
        except OSError:
            logger.warning("Failed to send macOS notification", exc_info=True)
