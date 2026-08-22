"""Component 4: System Supervisor.

The coordinator that manages lifecycle, state-machine transitions, and the
per-frame execution loop tying every interface-driven component together.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Type-checking only: importing this at runtime would make core depend on
    # implementations, inverting the dependency direction the rest of the
    # system keeps. The Supervisor only ever calls .write()/.close(), so any
    # object with that shape works.
    from src.implementations.feature_vector_csv_writer import FeatureVectorCsvWriter

from src.interfaces.alarm import AlarmEvent, IAlarmHandler
from src.interfaces.controller import IController, SystemStatus
from src.interfaces.detector import IAnomalyDetector
from src.interfaces.encoder import FeatureVector, IFeatureEncoder, IVideoEncoder
from src.interfaces.state import SupervisorState
from src.interfaces.video_source import IVideoSource

logger = logging.getLogger(__name__)

# Re-exported so existing `from src.core.supervisor import SupervisorState`
# imports keep working; the enum itself lives in the interfaces layer so that
# IController can reference it without depending on core.
__all__ = ["Supervisor", "SupervisorState"]


class Supervisor:
    """Runs the core video -> vectors -> features -> detection loop and owns
    the system's state machine.

    Any implementation of IVideoSource / IVideoEncoder / IFeatureEncoder /
    IAnomalyDetector / IAlarmHandler / IController can be swapped in via the
    constructor - the Supervisor only depends on the interfaces.

    Thread-safety: `latest_frame` and `state` are updated behind a lock, and
    everything an operator UI needs is pushed to the optional IController, so
    `run_forever()` can be driven from a background thread while a web server
    serves requests on the main thread.
    """

    def __init__(
        self,
        video_source: IVideoSource,
        video_encoder: IVideoEncoder,
        feature_encoder: IFeatureEncoder,
        anomaly_detector: IAnomalyDetector,
        alarm_handler: IAlarmHandler,
        collection_target_value: int = 1000,
        controller: IController | None = None,
        feature_csv_writer: "FeatureVectorCsvWriter | None" = None,
    ) -> None:
        self._video_source = video_source
        self._video_encoder = video_encoder
        self._feature_encoder = feature_encoder
        self._anomaly_detector = anomaly_detector
        self._alarm_handler = alarm_handler
        self._collection_target_value = collection_target_value
        self._controller = controller
        self._feature_csv_writer = feature_csv_writer

        self._training_buffer: list[list[FeatureVector]] = []
        self._processed_frame_count = 0
        self._total_alarms = 0

        self._lock = threading.Lock()
        self._state = SupervisorState.INITIALIZING
        self._latest_frame: np.ndarray | None = None

    # --- Public, thread-safe accessors ------------------------------------------------

    @property
    def state(self) -> SupervisorState:
        with self._lock:
            return self._state

    @property
    def latest_frame(self) -> np.ndarray | None:
        """The most recent annotated frame (detection overlays drawn on)."""
        with self._lock:
            return self._latest_frame

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
        """Snapshot the Supervisor's public state for the controller/UI."""
        return SystemStatus(
            state=self.state,
            processed_frame_count=self._processed_frame_count,
            collection_progress=self.collection_progress,
            collected_count=len(self._training_buffer),
            collection_target=self._collection_target_value,
            total_alarms=self._total_alarms,
        )

    def _set_state(self, new_state: SupervisorState) -> None:
        with self._lock:
            if new_state != self._state:
                logger.info("State transition: %s -> %s", self._state.name, new_state.name)
            self._state = new_state

    # --- Lifecycle -------------------------------------------------------------------

    def initialize(self) -> None:
        """Move from INITIALIZING into COLLECTING_DATA."""
        self._set_state(SupervisorState.COLLECTING_DATA)

    def run_forever(self) -> None:
        """Run the core loop until the video source is exhausted or an error occurs.

        Intended to be run on a background thread when a Controller is
        serving a UI on the main thread.
        """
        self.initialize()
        try:
            while True:
                should_continue = self.step()
                if not should_continue:
                    break
        finally:
            self._video_source.release()
            if self._feature_csv_writer is not None:
                self._feature_csv_writer.close()

    def step(self) -> bool:
        """Execute a single iteration of the core loop.

        Returns:
            False if the video source has no more frames (caller should stop
            calling step()), True otherwise.
        """
        success, frame = self._video_source.get_frame()
        if not success or frame is None:
            logger.warning("Video source returned no frame; stopping.")
            return False

        self._processed_frame_count += 1

        try:
            encoded = self._video_encoder.encode(frame)

            # The annotated frame is what the UI displays; the ML pipeline
            # only ever sees encoded.states.
            with self._lock:
                self._latest_frame = encoded.annotated_frame
            self._publish_frame(encoded.annotated_frame)

            features = self._feature_encoder.encode(encoded.states)
            self._write_feature_csv(features)
            self._handle_features(features)
        except Exception:
            logger.exception("Unhandled error while processing frame; entering ERROR state.")
            self._set_state(SupervisorState.ERROR)
            self._publish_status()
            raise

        self._publish_status()
        return True

    # --- Controller push helpers ------------------------------------------------------
    #
    # The Supervisor knows nothing about how (or whether) this data is
    # displayed - it just hands it to the IController implementation. A
    # misbehaving UI must never take down the processing loop, so every push
    # is guarded.

    def _publish_status(self) -> None:
        if self._controller is None:
            return
        try:
            self._controller.publish_status(self.build_status())
        except Exception:
            logger.exception("Controller.publish_status failed; continuing.")

    def _publish_frame(self, annotated_frame: np.ndarray) -> None:
        if self._controller is None:
            return
        try:
            self._controller.publish_frame(annotated_frame)
        except Exception:
            logger.exception("Controller.publish_frame failed; continuing.")

    def _write_feature_csv(self, features: list[FeatureVector]) -> None:
        """Observation-only side channel; must never disturb the pipeline."""
        if self._feature_csv_writer is None:
            return
        try:
            self._feature_csv_writer.write(features)
        except Exception:
            logger.exception("Feature CSV write failed; continuing.")

    def _publish_alarm(self, event: AlarmEvent) -> None:
        if self._controller is None:
            return
        try:
            self._controller.publish_alarm(event)
        except Exception:
            logger.exception("Controller.publish_alarm failed; continuing.")

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

        # TRAINING / INITIALIZING / ERROR: nothing to do with this frame's features.

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
        self._publish_status()
        try:
            self._anomaly_detector.fit(self._training_buffer)
        except Exception:
            logger.exception("Training failed; entering ERROR state.")
            self._set_state(SupervisorState.ERROR)
            raise
        self._set_state(SupervisorState.MONITORING)

    def _monitor(self, features: list[FeatureVector]) -> None:
        is_anomaly, score = self._anomaly_detector.predict(features)

        if is_anomaly:
            self._set_state(SupervisorState.ALARM_ACTIVE)
            self._total_alarms += 1
            event = AlarmEvent(
                timestamp=int(time.time() * 1000),
                vector=features,
                anomaly_score=score,
                description="Anomaly detected by IsolationForestAnomalyDetector.",
                frame=self.latest_frame,
            )
            self._alarm_handler.trigger_alarm(event)
            self._publish_alarm(event)
        else:
            self._set_state(SupervisorState.MONITORING)
