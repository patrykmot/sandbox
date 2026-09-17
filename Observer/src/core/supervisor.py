"""Component 4: System Supervisor.

The coordinator that manages lifecycle, state-machine transitions, and the
per-frame execution loop tying every interface-driven component together.

It knows nothing about a UI. Whatever displays the system *pulls* what it
needs - `build_status()` and `latest_frame` - and *requests* the three things
it may change. Those requests (START / STOP / select camera) are queued by
whatever thread calls them and applied here, at the top of a loop pass, so a
click in the browser never touches OpenCV or model state mid-frame.

Alarms go out through the injected IAlarmHandler, which is the one and only
way an event leaves this class.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    # Type-checking only: importing this at runtime would make core depend on
    # implementations, inverting the dependency direction the rest of the
    # system keeps. The Supervisor only ever calls .write()/.close(), so any
    # object with that shape works.
    from src.implementations.feature_vector_csv_writer import FeatureVectorCsvWriter

from src.interfaces.alarm import AlarmEvent, IAlarmHandler
from src.interfaces.controller import SystemStatus
from src.interfaces.detector import IAnomalyDetector
from src.interfaces.encoder import FeatureVector, IFeatureEncoder, IVideoEncoder
from src.interfaces.state import SupervisorState
from src.interfaces.video_source import IVideoSource

logger = logging.getLogger(__name__)

# Re-exported so existing `from src.core.supervisor import SupervisorState`
# imports keep working; the enum itself lives in the interfaces layer so that
# ISupervisorPort can reference it without depending on core.
__all__ = ["Supervisor", "SupervisorState"]

#: Pause between loop passes when there is no camera to read from (failed
#: open, or ERROR waiting for STOP). Long enough not to spin a core, short
#: enough that a STOP feels immediate.
_WAIT_SECONDS = 0.5


class Supervisor:
    """Runs the core video -> vectors -> features -> detection loop and owns
    the system's state machine.

    Any implementation of IVideoSource / IVideoEncoder / IFeatureEncoder /
    IAnomalyDetector / IAlarmHandler can be swapped in via the constructor -
    the Supervisor only depends on the interfaces. It satisfies
    ISupervisorPort structurally, without importing or inheriting anything
    from the UI side.

    Video sources are built through a factory rather than handed in ready-made,
    because the operator picks the camera at runtime: the Supervisor opens,
    releases and reopens sources as the selection changes.

    Thread-safety: mutable state is updated behind a lock, commands arrive on
    a queue, and every read a UI needs is a lock-guarded snapshot - so
    `run_forever()` can be driven from a background thread while a web server
    serves requests on the main thread.
    """

    def __init__(
        self,
        video_source_factory: Callable[[int | str], IVideoSource],
        video_encoder: IVideoEncoder,
        feature_encoder: IFeatureEncoder,
        anomaly_detector: IAnomalyDetector,
        alarm_handler: IAlarmHandler,
        collection_target_value: int = 1000,
        feature_csv_writer: "FeatureVectorCsvWriter | None" = None,
        camera: int | str = 0,
    ) -> None:
        self._video_source_factory = video_source_factory
        self._video_encoder = video_encoder
        self._feature_encoder = feature_encoder
        self._anomaly_detector = anomaly_detector
        self._alarm_handler = alarm_handler
        self._collection_target_value = collection_target_value
        self._feature_csv_writer = feature_csv_writer

        self._training_buffer: list[list[FeatureVector]] = []
        self._processed_frame_count = 0
        self._total_alarms = 0
        self._run_id = 0

        self._commands: queue.Queue[tuple[str, object]] = queue.Queue()
        self._shutdown = threading.Event()

        self._lock = threading.Lock()
        self._state = SupervisorState.INITIALIZING
        self._latest_frame: np.ndarray | None = None
        self._source: IVideoSource | None = None
        self._camera: int | str = camera
        self._error: str | None = None

    # --- Public, thread-safe accessors ------------------------------------------------

    @property
    def state(self) -> SupervisorState:
        with self._lock:
            return self._state

    @property
    def latest_frame(self) -> np.ndarray | None:
        """The most recent frame shown to the UI (annotated while scanning,
        raw camera preview while IDLE)."""
        with self._lock:
            return self._latest_frame

    @property
    def camera(self) -> int | str:
        """The currently selected video source id."""
        with self._lock:
            return self._camera

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def collection_progress(self) -> float:
        """Fraction (0.0-1.0) of the COLLECTING_DATA target reached so far."""
        if self._collection_target_value <= 0:
            return 1.0
        return min(1.0, len(self._training_buffer) / self._collection_target_value)

    @property
    def processed_frame_count(self) -> int:
        return self._processed_frame_count

    @property
    def total_alarms(self) -> int:
        return self._total_alarms

    def build_status(self) -> SystemStatus:
        """Snapshot the Supervisor's public state for whatever is displaying it.

        Called from another thread now that the UI pulls rather than being
        pushed to, so every field is read in one lock acquisition - a status
        that mixes a new state with an old frame count would be a lie.
        """
        with self._lock:
            state, camera, error = self._state, self._camera, self._error
            processed = self._processed_frame_count
            total_alarms = self._total_alarms
            run_id = self._run_id
            collected = len(self._training_buffer)

        target = self._collection_target_value
        progress = 1.0 if target <= 0 else min(1.0, collected / target)

        return SystemStatus(
            state=state,
            processed_frame_count=processed,
            collection_progress=progress,
            collected_count=collected,
            collection_target=target,
            total_alarms=total_alarms,
            camera=camera,
            error=error,
            run_id=run_id,
        )

    def _set_state(self, new_state: SupervisorState) -> None:
        with self._lock:
            if new_state != self._state:
                logger.info("State transition: %s -> %s", self._state.name, new_state.name)
            self._state = new_state

    def _set_error(self, message: str | None) -> None:
        with self._lock:
            self._error = message
        if message:
            logger.error("%s", message)

    # --- Commands (ISupervisorPort - safe to call from any thread) --------------------

    def request_start(self) -> None:
        self._commands.put(("start", None))

    def request_stop(self) -> None:
        self._commands.put(("stop", None))

    def request_camera(self, camera: int | str) -> None:
        self._commands.put(("camera", camera))

    def shutdown(self) -> None:
        """End `run_forever()` after the current pass (process teardown)."""
        self._shutdown.set()

    def _drain_commands(self) -> None:
        while True:
            try:
                name, payload = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._apply_command(name, payload)
            except Exception:
                logger.exception("Command %r failed; continuing.", name)

    def _apply_command(self, name: str, payload: object) -> None:
        if name == "start":
            self._do_start()
        elif name == "stop":
            self._do_stop()
        elif name == "camera":
            self._do_select_camera(payload)  # type: ignore[arg-type]
        else:
            logger.warning("Unknown command %r ignored.", name)

    def _do_start(self) -> None:
        """IDLE -> COLLECTING_DATA, with everything learned so far discarded."""
        if self.state != SupervisorState.IDLE:
            logger.info("START ignored: only valid from IDLE.")
            return

        self._reset_run_state()
        if not self._ensure_source():
            return  # stays IDLE; _ensure_source set the error message

        # A new baseline means the previous run's alarms no longer mean
        # anything. Bumping the run id says so; nobody has to be told to
        # forget them. Deliberately not in _reset_run_state(), which STOP
        # also calls - STOP keeps the panel.
        with self._lock:
            self._run_id += 1

        self._set_error(None)
        self._set_state(SupervisorState.COLLECTING_DATA)

    def _do_stop(self) -> None:
        """Anything -> IDLE. The camera stays open so the preview continues."""
        if self.state == SupervisorState.IDLE:
            return
        self._reset_run_state()
        self._set_error(None)
        self._set_state(SupervisorState.IDLE)

    def _do_select_camera(self, camera: int | str) -> None:
        if self.state != SupervisorState.IDLE:
            logger.info("Camera change ignored: only valid from IDLE.")
            return
        if camera == self.camera and self._source is not None:
            return
        self._close_source()
        with self._lock:
            self._camera = camera
        self._set_error(None)
        logger.info("Camera selected: %r", camera)
        # Reopened by the next preview pass.

    def _reset_run_state(self) -> None:
        """Forget the baseline, the counters and any per-object tracking."""
        self._training_buffer = []
        self._processed_frame_count = 0
        self._total_alarms = 0
        try:
            self._video_encoder.reset()
        except Exception:
            logger.exception("Video encoder reset failed; continuing.")

    # --- Video source -----------------------------------------------------------------

    def _ensure_source(self) -> bool:
        """Open the selected camera if it isn't already. False on failure."""
        if self._source is not None:
            return True
        camera = self.camera
        try:
            self._source = self._video_source_factory(camera)
        except Exception as exc:
            self._source = None
            self._set_error(f"Could not open camera {camera!r}: {exc}")
            return False
        self._set_error(None)
        return True

    def _close_source(self) -> None:
        if self._source is None:
            return
        try:
            self._source.release()
        except Exception:
            logger.exception("Releasing the video source failed; continuing.")
        finally:
            self._source = None

    # --- Lifecycle -------------------------------------------------------------------

    def initialize(self) -> None:
        """Move from INITIALIZING into IDLE, waiting for START."""
        self._set_state(SupervisorState.IDLE)

    def run_forever(self) -> None:
        """Run until `shutdown()` is called.

        Unlike a plain processing loop this never ends by itself: STOP and
        ERROR both return to a waiting state rather than falling out, because
        the dashboard has to stay useful after either.

        Intended to be run on a background thread while a Controller serves a
        UI on the main thread.
        """
        self.initialize()
        try:
            while not self._shutdown.is_set():
                self._drain_commands()
                self.tick()
        finally:
            self._close_source()
            if self._feature_csv_writer is not None:
                self._feature_csv_writer.close()

    def tick(self) -> None:
        """One pass of the loop, whatever state the system is in."""
        state = self.state

        if state == SupervisorState.IDLE:
            self._preview()
            return

        if state == SupervisorState.ERROR:
            # Nothing to do until the operator presses STOP.
            self._shutdown.wait(_WAIT_SECONDS)
            return

        self.step()

    def _preview(self) -> None:
        """IDLE: show the selected camera without detecting anything.

        The point is to let the operator see what the camera sees before
        committing to a scan, so this deliberately skips the encoder, the
        feature pipeline and the detector - it is a viewfinder, not a run.
        """
        if not self._ensure_source():
            self._shutdown.wait(_WAIT_SECONDS)
            return

        assert self._source is not None
        success, frame = self._source.get_frame()
        if not success or frame is None:
            self._close_source()
            self._set_error(f"Camera {self.camera!r} stopped delivering frames.")
            self._shutdown.wait(_WAIT_SECONDS)
            return

        with self._lock:
            self._latest_frame = frame

    def step(self) -> bool:
        """Execute a single iteration of the scanning loop.

        Returns:
            False if no frame could be read (the system moves to ERROR),
            True otherwise.
        """
        if not self._ensure_source():
            self._set_state(SupervisorState.ERROR)
            return False

        assert self._source is not None
        success, frame = self._source.get_frame()
        if not success or frame is None:
            logger.warning("Video source returned no frame; entering ERROR state.")
            self._close_source()
            self._set_error("Video source stopped delivering frames.")
            self._set_state(SupervisorState.ERROR)
            return False

        with self._lock:
            self._processed_frame_count += 1

        try:
            encoded = self._video_encoder.encode(frame)

            # The annotated frame is what the UI displays; the ML pipeline
            # only ever sees encoded.states.
            with self._lock:
                self._latest_frame = encoded.annotated_frame

            features = self._feature_encoder.encode(encoded.states)
            self._write_feature_csv(features)
            self._handle_features(features)
        except Exception as exc:
            logger.exception("Unhandled error while processing frame; entering ERROR state.")
            self._set_error(f"Processing failed: {exc}")
            self._set_state(SupervisorState.ERROR)
            return False

        return True

    # --- Side channels ----------------------------------------------------------------

    def _write_feature_csv(self, features: list[FeatureVector]) -> None:
        """Observation-only side channel; must never disturb the pipeline."""
        if self._feature_csv_writer is None:
            return
        try:
            self._feature_csv_writer.write(features)
        except Exception:
            logger.exception("Feature CSV write failed; continuing.")

    # --- Core state-dependent branching -----------------------------------------------

    def _handle_features(self, features: list[FeatureVector]) -> None:
        current_state = self.state

        if current_state == SupervisorState.COLLECTING_DATA:
            self._training_buffer.append(features)
            target_reached = len(self._training_buffer) >= self._collection_target_value
            self._print_collection_progress(final=target_reached)
            if target_reached:
                self._train()
            return

        if current_state in (SupervisorState.MONITORING, SupervisorState.ALARM_ACTIVE):
            self._monitor(features)
            return

        # TRAINING / INITIALIZING / IDLE / ERROR: nothing to do with this
        # frame's features.

    def _print_collection_progress(self, final: bool, bar_width: int = 20) -> None:
        """Print a single, in-place-updating progress line to the console
        while COLLECTING_DATA is in progress.

        Uses a bare carriage return (no logging - log lines carry a
        timestamp/level prefix and start a new one each call) so repeated
        calls overwrite the same terminal line instead of scrolling. The
        final call of the collection phase ends with a real newline so the
        next (logged) TRAINING/MONITORING lines start cleanly.
        """
        count = len(self._training_buffer)
        target = self._collection_target_value
        progress = 1.0 if target <= 0 else min(1.0, count / target)

        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)
        line = f"\rCollecting baseline data: [{bar}] {progress * 100:5.1f}% ({count}/{target})"

        sys.stdout.write(line)
        sys.stdout.write("\n" if final else "")
        sys.stdout.flush()

    def _train(self) -> None:
        self._set_state(SupervisorState.TRAINING)
        try:
            self._anomaly_detector.fit(self._training_buffer)
        except Exception as exc:
            logger.exception("Training failed; entering ERROR state.")
            self._set_error(f"Training failed: {exc}")
            self._set_state(SupervisorState.ERROR)
            return
        self._set_state(SupervisorState.MONITORING)

    def _monitor(self, features: list[FeatureVector]) -> None:
        is_anomaly, score = self._anomaly_detector.predict(features)

        if is_anomaly:
            self._set_state(SupervisorState.ALARM_ACTIVE)
            with self._lock:
                self._total_alarms += 1
            event = AlarmEvent(
                timestamp=int(time.time() * 1000),
                vector=features,
                anomaly_score=score,
                description="Anomaly detected by IsolationForestAnomalyDetector.",
                frame=self.latest_frame,
            )
            # The one way an event leaves this class. Fan-out and per-sink
            # isolation are the handler's job (see CompositeAlarmHandler).
            self._alarm_handler.trigger_alarm(event)
        else:
            self._set_state(SupervisorState.MONITORING)
