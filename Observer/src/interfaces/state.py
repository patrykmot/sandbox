"""Shared system-state enum.

Lives in `interfaces` rather than `core` so that interface definitions
(notably IController) can reference the system state without importing from
the core layer - core depends on interfaces, never the other way around.
`src.core.supervisor` re-exports SupervisorState for backwards compatibility.
"""

from __future__ import annotations

from enum import Enum, auto


class SupervisorState(Enum):
    INITIALIZING = auto()
    COLLECTING_DATA = auto()
    TRAINING = auto()
    MONITORING = auto()
    ALARM_ACTIVE = auto()
    ERROR = auto()
