"""Interface 7: the port between the Supervisor and whatever drives it.

The Supervisor owns no UI knowledge and pushes nothing. A UI - a FastAPI
dashboard, a TUI, a metrics exporter - *pulls* what it wants to show, when it
wants to show it, and *requests* the three things it may change. Both
directions are the one Protocol below.

Reads (`build_status`, `latest_frame`) are snapshots taken under the
Supervisor's lock. Requests are queued and applied by the Supervisor on its
own thread, at a frame boundary, so a web request never touches OpenCV or
model state mid-frame. Neither blocks on frame processing, so both are safe
from any thread.

Alarms are *not* here: they are events, not state, and travel through
IAlarmHandler (src/interfaces/alarm.py) like any other alarm sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

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
        detector: Id of the anomaly-detector implementation this run uses.
        training_progress: TRAINING progress as a 0.0-1.0 fraction. Reset to
            0.0 by START; meaningless outside TRAINING.
        run_id: Incremented on every successful START. Identifies the baseline
            a given alarm was scored against, so a UI can tell this run's
            alarms from a previous run's without being told to forget them.
    """

    state: SupervisorState
    processed_frame_count: int
    collection_progress: float
    collected_count: int
    collection_target: int
    total_alarms: int
    camera: int | str | None = None
    error: str | None = None
    detector: str = ""
    training_progress: float = 0.0
    run_id: int = 0


@runtime_checkable
class ISupervisorPort(Protocol):
    """What a UI may read from, and ask of, the Supervisor.

    Structural, not inherited: anything with this shape works, and the
    Supervisor satisfies it without importing anything from the UI side.
    """

    @property
    def latest_frame(self) -> np.ndarray | None:
        """The most recent frame to display - the raw camera preview while
        IDLE, the annotated frame with detection overlays once scanning."""
        ...

    def build_status(self) -> SystemStatus:
        """Snapshot the Supervisor's public state."""
        ...

    def request_start(self) -> None:
        """Begin collecting a baseline on the selected camera (IDLE only)."""
        ...

    def request_stop(self) -> None:
        """Return to IDLE from any state, discarding what was learned."""
        ...

    def request_camera(self, camera: int | str) -> None:
        """Select a different video source (IDLE only)."""
        ...

    def request_detector(self, detector: str) -> None:
        """Select a different anomaly-detector implementation (IDLE only)."""
        ...
