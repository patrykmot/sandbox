"""Implementation 2 of IAlarmHandler: fan one alarm out to several sinks.

The Supervisor takes exactly one IAlarmHandler, but an alarm usually has more
than one audience - the console, the dashboard, later perhaps a webhook. This
is how they are combined.

It is also where the isolation lives. The Supervisor calls trigger_alarm from
inside the frame loop's try block, so an exception raised by a sink would drop
the whole system into ERROR. A dashboard hiccup must never do that, so every
sink is called behind its own guard: one failing sink is logged, and the rest
still fire.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from src.interfaces.alarm import AlarmEvent, IAlarmHandler

logger = logging.getLogger(__name__)


class CompositeAlarmHandler(IAlarmHandler):
    """Dispatches each AlarmEvent to every handler it was given, in order."""

    def __init__(self, handlers: Iterable[IAlarmHandler]) -> None:
        self._handlers: tuple[IAlarmHandler, ...] = tuple(handlers)

    def trigger_alarm(self, event: AlarmEvent) -> None:
        for handler in self._handlers:
            try:
                handler.trigger_alarm(event)
            except Exception:
                logger.exception(
                    "Alarm sink %s failed; continuing with the remaining sinks.",
                    type(handler).__name__,
                )
