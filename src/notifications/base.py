from __future__ import annotations

from typing import Protocol

from src.core.state_machine import State


class Notifier(Protocol):
    def notify(self, new_state: State, old_state: State) -> None: ...
