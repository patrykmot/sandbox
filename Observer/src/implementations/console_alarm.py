"""Implementation 1 of IAlarmHandler: high-visibility console logging."""

from __future__ import annotations

import sys
from collections import deque
from datetime import datetime, timezone

from src.interfaces.alarm import AlarmEvent, IAlarmHandler

_BANNER = "!" * 70


class ConsoleLoggerAlarmHandler(IAlarmHandler):
    """Logs alarms loudly to stderr and keeps recent history in memory.

    The in-memory history (`recent_alarms`) is exposed for a future Web UI
    (Phase II) to read without needing a separate storage layer.
    """

    def __init__(self, history_size: int = 100) -> None:
        self.recent_alarms: deque[AlarmEvent] = deque(maxlen=history_size)

    def trigger_alarm(self, event: AlarmEvent) -> None:
        self.recent_alarms.append(event)

        readable_time = datetime.fromtimestamp(
            event.timestamp / 1000.0, tz=timezone.utc
        ).isoformat()

        # message = (
        #     f"\n{_BANNER}\n"
        #     f"  ANOMALY DETECTED\n"
        #     f"  Time:        {readable_time}\n"
        #     f"  Score:       {event.anomaly_score:.4f}\n"
        #     f"  Description: {event.description}\n"
        #     f"  Objects:     {len(event.vector)}\n"
        #     f"{_BANNER}\n"
        # )
        # print(message, file=sys.stderr)
