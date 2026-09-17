"""Shared system-state enum.

Lives in `interfaces` rather than `core` so that interface definitions
(notably ISupervisorPort) can reference the system state without importing from
the core layer - core depends on interfaces, never the other way around.
`src.core.supervisor` re-exports SupervisorState for backwards compatibility.

State machine
-------------

    INITIALIZING -> IDLE -> COLLECTING_DATA -> TRAINING -> MONITORING
                      ^                                        |
                      |                                   ALARM_ACTIVE
                      |                                        |
                      +--------------- STOP -------------------+

IDLE is the resting state: the selected camera is open and previewed, but
nothing is detected, collected or scored. START moves to COLLECTING_DATA;
STOP returns to IDLE from anywhere, including ERROR, which is why the UI
needs only one button.
"""

from __future__ import annotations

from enum import Enum, auto


class SupervisorState(Enum):
    INITIALIZING = auto()
    IDLE = auto()
    COLLECTING_DATA = auto()
    TRAINING = auto()
    MONITORING = auto()
    ALARM_ACTIVE = auto()
    ERROR = auto()
