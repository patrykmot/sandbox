"""Interface 7: Controller (presentation / operator-facing layer) and the
narrow command channel back from it.

The Supervisor owns no UI knowledge. It simply *pushes* everything an
operator might want to see into an IController implementation, which decides
how (or whether) to present it - a FastAPI dashboard, a TUI, a metrics
exporter, or a no-op in headless runs.

Commands travel the other way through IControlTarget, which the Supervisor
implements. A controller may only *request* - every request is queued and
applied by the Supervisor on its own thread, at a frame boundary, so a web
request never touches OpenCV or model state mid-frame.
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
        processed_frame_count: Total frames processed since the last START.
        collection_progress: COLLECTING_DATA progress as a 0.0-1.0 fraction.
        collected_count: Number of frames collected so far.
        collection_target: Target number of frames to collect.
        total_alarms: Total alarms raised since the last START.
        camera: Id of the currently selected video source, if any.
        error: Last error message, cleared on the next successful START/STOP.
    """

    state: SupervisorState
    processed_frame_count: int
    collection_progress: float
    collected_count: int
    collection_target: int
    total_alarms: int
    camera: int | str | None = None
    error: str | None = None


class IControlTarget(ABC):
    """What a controller may ask the Supervisor to do.

    Deliberately tiny: two verbs plus source selection. Implementations queue
    the request rather than acting on it, so these are safe to call from any
    thread and never block the caller.
    """

    @abstractmethod
    def request_start(self) -> None:
        """Begin collecting a baseline on the selected camera (IDLE only)."""
        raise NotImplementedError

    @abstractmethod
    def request_stop(self) -> None:
        """Return to IDLE from any state, discarding what was learned."""
        raise NotImplementedError

    @abstractmethod
    def request_camera(self, camera: int | str) -> None:
        """Select a different video source (IDLE only)."""
        raise NotImplementedError


class IController(ABC):
    """Abstract interface for presenting Supervisor state to an operator."""

    @abstractmethod
    def publish_status(self, status: SystemStatus) -> None:
        """Receive the latest system status snapshot (called every frame)."""
        raise NotImplementedError

    @abstractmethod
    def publish_frame(self, annotated_frame: np.ndarray) -> None:
        """Receive the latest frame for live display.

        While IDLE this is the raw camera preview; once scanning it is the
        annotated frame with detection overlays.
        """
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

    def clear_alarms(self) -> None:
        """Drop any displayed alarm history.

        Called when a new scan starts: alarms scored against the previous
        baseline mean nothing against the new one. Not abstract - a
        controller with nothing to clear can ignore it.
        """
        return None
