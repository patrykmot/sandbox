"""End-to-end test of the state machine + detector pipeline using fakes.

No camera, YOLO, or OpenCV required: a fake IVideoSource yields synthetic
frames and a fake IVideoEncoder yields synthetic StateVectors directly, so
this exercises DummyFeatureEncoder, IsolationForestAnomalyDetector,
ConsoleLoggerAlarmHandler, and the full Supervisor state machine for real.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.supervisor import Supervisor, SupervisorState
from src.implementations.console_alarm import ConsoleLoggerAlarmHandler
from src.implementations.dummy_feature_encoder import DummyFeatureEncoder
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector
from src.interfaces.encoder import IVideoEncoder, StateVector
from src.interfaces.video_source import IVideoSource

COLLECTION_TARGET = 60


class FakeVideoSource(IVideoSource):
    """Yields a fixed number of blank frames, then signals end-of-stream."""

    def __init__(self, frame_count: int) -> None:
        self._remaining = frame_count
        self.released = False

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FakeVideoEncoder(IVideoEncoder):
    """Produces one synthetic "normal" object per frame, except for a single
    injected outlier frame with an extreme position/size to trigger an alarm.
    """

    def __init__(self, outlier_frame_index: int) -> None:
        self._frame_index = -1
        self._outlier_frame_index = outlier_frame_index
        self._rng = random.Random(42)

    def encode(self, frame: np.ndarray) -> list[StateVector]:
        self._frame_index += 1
        t_ms = 1_700_000_000_000 + self._frame_index * 33

        if self._frame_index == self._outlier_frame_index:
            return [
                StateVector(
                    t=t_ms, x=9999.0, y=9999.0, vx=5000.0, vy=5000.0,
                    size=999999.0, object_type=0,
                )
            ]

        return [
            StateVector(
                t=t_ms,
                x=100.0 + self._rng.uniform(-2, 2),
                y=100.0 + self._rng.uniform(-2, 2),
                vx=self._rng.uniform(-1, 1),
                vy=self._rng.uniform(-1, 1),
                size=500.0 + self._rng.uniform(-5, 5),
                object_type=0,
            )
        ]


def build_test_supervisor(total_frames: int, outlier_frame_index: int) -> tuple[Supervisor, FakeVideoSource]:
    video_source = FakeVideoSource(total_frames)
    video_encoder = FakeVideoEncoder(outlier_frame_index=outlier_frame_index)
    feature_encoder = DummyFeatureEncoder()
    anomaly_detector = IsolationForestAnomalyDetector(contamination=0.05)
    alarm_handler = ConsoleLoggerAlarmHandler()

    supervisor = Supervisor(
        video_source=video_source,
        video_encoder=video_encoder,
        feature_encoder=feature_encoder,
        anomaly_detector=anomaly_detector,
        alarm_handler=alarm_handler,
        collection_target_value=COLLECTION_TARGET,
    )
    return supervisor, video_source


def test_reaches_monitoring_after_collection_target() -> None:
    # +1 frame after the collection target so we can observe MONITORING.
    supervisor, _ = build_test_supervisor(
        total_frames=COLLECTION_TARGET + 1, outlier_frame_index=-1
    )
    supervisor.initialize()
    assert supervisor.state == SupervisorState.COLLECTING_DATA

    for _ in range(COLLECTION_TARGET):
        assert supervisor.step() is True

    assert supervisor.state == SupervisorState.MONITORING

    assert supervisor.step() is True
    assert supervisor.state == SupervisorState.MONITORING
    print("OK: test_reaches_monitoring_after_collection_target")


def test_outlier_triggers_alarm() -> None:
    outlier_index = COLLECTION_TARGET + 3
    supervisor, _ = build_test_supervisor(
        total_frames=outlier_index + 1, outlier_frame_index=outlier_index
    )
    supervisor.initialize()

    for _ in range(outlier_index):
        assert supervisor.step() is True
    assert supervisor.state == SupervisorState.MONITORING

    # This step processes the injected outlier frame.
    assert supervisor.step() is True
    assert supervisor.state == SupervisorState.ALARM_ACTIVE

    alarm_handler = supervisor._alarm_handler  # test-only introspection
    assert len(alarm_handler.recent_alarms) == 1
    assert alarm_handler.recent_alarms[0].anomaly_score < 0
    print("OK: test_outlier_triggers_alarm")


def test_video_source_exhaustion_stops_loop() -> None:
    supervisor, video_source = build_test_supervisor(
        total_frames=5, outlier_frame_index=-1
    )
    supervisor.run_forever()
    assert video_source.released is True
    assert supervisor.processed_frame_count == 5
    print("OK: test_video_source_exhaustion_stops_loop")


if __name__ == "__main__":
    test_reaches_monitoring_after_collection_target()
    test_outlier_triggers_alarm()
    test_video_source_exhaustion_stops_loop()
    print("\nAll tests passed.")
