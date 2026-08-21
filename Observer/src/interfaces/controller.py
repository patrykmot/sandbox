"""Interface 7: Controller (presentation / operator-facing layer).

The Supervisor owns no UI knowledge. It simply *pushes* everything an
operator might want to see into an IController implementation, which decides
how (or whether) to present it - a FastAPI dashboard, a TUI, a metrics
exporter, or a no-op in headless runs.

Future scope (deliberately NOT implemented yet, per the Phase II spec): an
IController will also be able to signal actions back to the Supervisor -
pause, restart, stop training. The interface is intentionally one-directional
for now so that adding that later is additive rather than a redesign.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.interfaces.alarm import AlarmEvent
from src.interfaces.state import SupervisorState


@dataclass(frozen=True, slots=True)
class SystemStatus:
    """An immutable snapshot of the Supervisor's public state.

    Attributes:
        state: Current state-machine state.
        processed_frame_count: Total frames processed since startup.
        collection_progress: COLLECTING_DATA progress as a 0.0-1.0 fraction.
        collected_count: Number of frames collected so far.
        collection_target: Target number of frames to collect.
        total_alarms: Total alarms raised since startup.
    """

    state: SupervisorState
    processed_frame_count: int
    collection_progress: float
    collected_count: int
    collection_target: int
    total_alarms: int


class IController(ABC):
    """Abstract interface for presenting Supervisor state to an operator."""

    @abstractmethod
    def publish_status(self, status: SystemStatus) -> None:
        """Receive the latest system status snapshot (called every frame)."""
        raise NotImplementedError

    @abstractmethod
    def publish_frame(self, annotated_frame: np.ndarray) -> None:
        """Receive the latest annotated frame for live display."""
        raise NotImplementedError

    @abstractmethod
    def publish_alarm(self, event: AlarmEvent) -> None:
        """Receive a newly raised alarm, including its frame snapshot."""
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """Start serving and block until shutdown.

        Called from the main thread; the Supervisor loop runs on its own
        background thread so this may block indefinitely.
        """
        raise NotImplementedError
