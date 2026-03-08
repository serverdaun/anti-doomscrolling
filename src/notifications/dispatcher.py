from __future__ import annotations

from src.core.state_machine import State
from src.notifications.base import Notifier


class NotificationDispatcher:
    def __init__(self) -> None:
        self._notifiers: list[Notifier] = []

    def add(self, notifier: Notifier) -> None:
        self._notifiers.append(notifier)

    def on_state_change(self, new_state: State, old_state: State) -> None:
        for notifier in self._notifiers:
            notifier.notify(new_state, old_state)
