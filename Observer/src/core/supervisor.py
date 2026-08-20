"""Component 4: System Supervisor.

The coordinator that manages lifecycle, state-machine transitions, and the
per-frame execution loop tying every interface-driven component together.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto

import numpy as np

from src.interfaces.alarm import AlarmEvent, IAlarmHandler
from src.interfaces.detector import IAnomalyDetector
from src.interfaces.encoder import FeatureVector, IFeatureEncoder, IVideoEncoder
from src.interfaces.video_source import IVideoSource

logger = logging.getLogger(__name__)


class SupervisorState(Enum):
    INITIALIZING = auto()
    COLLECTING_DATA = auto()
    TRAINING = auto()
    MONITORING = auto()
    ALARM_ACTIVE = auto()
    ERROR = auto()


class Supervisor:
    """Runs the core video -> vectors -> features -> detection loop and owns
    the system's state machine.

    Any implementation of IVideoSource / IVideoEncoder / IFeatureEncoder /
    IAnomalyDetector / IAlarmHandler can be swapped in via the constructor -
    the Supervisor only depends on the interfaces.

    Thread-safety: `latest_frame` and `state` are updated behind a lock so a
    future Web UI thread (Phase II) can safely read them while `run()` keeps
    looping in the background. Phase I itself runs `run()` synchronously on
    the calling thread.
    """

    def __init__(
        self,
        video_source: IVideoSource,
        video_encoder: IVideoEncoder,
        feature_encoder: IFeatureEncoder,
        anomaly_detector: IAnomalyDetector,
        alarm_handler: IAlarmHandler,
        collection_target_value: int = 1000,
    ) -> None:
        self._video_source = video_source
        self._video_encoder = video_encoder
        self._feature_encoder = feature_encoder
        self._anomaly_detector = anomaly_detector
        self._alarm_handler = alarm_handler
        self._collection_target_value = collection_target_value

        self._training_buffer: list[list[FeatureVector]] = []
        self._processed_frame_count = 0

        self._lock = threading.Lock()
        self._state = SupervisorState.INITIALIZING
        self._latest_frame: np.ndarray | None = None

    # --- Public, thread-safe accessors (for a future Web UI) -----------------------

    @property
    def state(self) -> SupervisorState:
        with self._lock:
            return self._state

    @property
    def latest_frame(self) -> np.ndarray | None:
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
        """Run the core loop until the video source is exhausted or an error occurs."""
        self.initialize()
        try:
            while True:
                should_continue = self.step()
                if not should_continue:
                    break
        finally:
            self._video_source.release()

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

        with self._lock:
            self._latest_frame = frame
        self._processed_frame_count += 1

        try:
            states = self._video_encoder.encode(frame)
            features = self._feature_encoder.encode(states)
            self._handle_features(features)
        except Exception:
            logger.exception("Unhandled error while processing frame; entering ERROR state.")
            self._set_state(SupervisorState.ERROR)
            raise

        return True

    # --- Core state-dependent branching -----------------------------------------------

    def _handle_features(self, features: list[FeatureVector]) -> None:
        current_state = self.state

        if current_state == SupervisorState.COLLECTING_DATA:
            self._training_buffer.append(features)
            logger.debug(
                "Collected frame %d/%d",
                len(self._training_buffer),
                self._collection_target_value,
            )
            if len(self._training_buffer) >= self._collection_target_value:
                self._train()
            return

        if current_state in (SupervisorState.MONITORING, SupervisorState.ALARM_ACTIVE):
            self._monitor(features)
            return

        # TRAINING / INITIALIZING / ERROR: nothing to do with this frame's features.

    def _train(self) -> None:
        self._set_state(SupervisorState.TRAINING)
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
            event = AlarmEvent(
                timestamp=int(time.time() * 1000),
                vector=features,
                anomaly_score=score,
                description="Anomaly detected by IsolationForestAnomalyDetector.",
                frame=self.latest_frame,
            )
            self._alarm_handler.trigger_alarm(event)
        else:
            self._set_state(SupervisorState.MONITORING)
