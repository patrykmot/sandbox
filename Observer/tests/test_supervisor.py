"""End-to-end test of the state machine + detector pipeline using mocks.

Per project_prompt.md's testing requirement: no camera is needed to run
these. IVideoSource and IVideoEncoder - the two interfaces that would
otherwise require real hardware/a GPU - are mocked with
unittest.mock.create_autospec. DummyFeatureEncoder,
IsolationForestAnomalyDetector, and ConsoleLoggerAlarmHandler are exercised
for real, since mocking those would just test the mocks instead of the
actual anomaly-detection logic.
"""

from __future__ import annotations

import random
from unittest.mock import create_autospec

import numpy as np
import pytest

from src.core.supervisor import Supervisor, SupervisorState
from src.implementations.console_alarm import ConsoleLoggerAlarmHandler
from src.implementations.dummy_feature_encoder import DummyFeatureEncoder
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector
from src.interfaces.encoder import IVideoEncoder, StateVector
from src.interfaces.video_source import IVideoSource

COLLECTION_TARGET = 60
BLANK_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def make_normal_states(count: int, start_t_ms: int = 1_700_000_000_000) -> list[list[StateVector]]:
    """One synthetic "normal" object per frame, mimicking a stationary target
    with small jitter - the same distribution IsolationForest is trained on.
    """
    rng = random.Random(42)
    frames: list[list[StateVector]] = []
    for i in range(count):
        t_ms = start_t_ms + i * 33
        frames.append(
            [
                StateVector(
                    t=t_ms,
                    x=100.0 + rng.uniform(-2, 2),
                    y=100.0 + rng.uniform(-2, 2),
                    vx=rng.uniform(-1, 1),
                    vy=rng.uniform(-1, 1),
                    size=500.0 + rng.uniform(-5, 5),
                    object_type=0,
                )
            ]
        )
    return frames


def make_outlier_state(t_ms: int) -> list[StateVector]:
    return [
        StateVector(t=t_ms, x=9999.0, y=9999.0, vx=5000.0, vy=5000.0, size=999999.0, object_type=0)
    ]


def build_mock_video_source(frame_count: int) -> IVideoSource:
    """A mocked IVideoSource yielding `frame_count` blank frames, then
    signaling end-of-stream (False, None) forever after.
    """
    mock_source = create_autospec(IVideoSource, instance=True)
    frames = [(True, BLANK_FRAME)] * frame_count
    mock_source.get_frame.side_effect = frames + [(False, None)] * 10
    return mock_source


def build_mock_video_encoder(state_sequence: list[list[StateVector]]) -> IVideoEncoder:
    """A mocked IVideoEncoder that returns each frame's StateVectors in order,
    regardless of the (unused, since detection is mocked) input frame.
    """
    mock_encoder = create_autospec(IVideoEncoder, instance=True)
    mock_encoder.encode.side_effect = state_sequence
    return mock_encoder


def build_supervisor(video_source: IVideoSource, video_encoder: IVideoEncoder) -> Supervisor:
    return Supervisor(
        video_source=video_source,
        video_encoder=video_encoder,
        feature_encoder=DummyFeatureEncoder(),
        anomaly_detector=IsolationForestAnomalyDetector(contamination=0.05),
        alarm_handler=ConsoleLoggerAlarmHandler(),
        collection_target_value=COLLECTION_TARGET,
    )


def test_reaches_monitoring_after_collection_target() -> None:
    # +1 frame after the collection target so we can observe MONITORING.
    total_frames = COLLECTION_TARGET + 1
    states = make_normal_states(total_frames)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder)

    supervisor.initialize()
    assert supervisor.state == SupervisorState.COLLECTING_DATA

    for _ in range(COLLECTION_TARGET):
        assert supervisor.step() is True
    assert supervisor.state == SupervisorState.MONITORING

    assert supervisor.step() is True
    assert supervisor.state == SupervisorState.MONITORING


def test_outlier_triggers_alarm() -> None:
    outlier_index = COLLECTION_TARGET + 3
    total_frames = outlier_index + 1

    states = make_normal_states(total_frames)
    states[outlier_index] = make_outlier_state(states[outlier_index][0].t)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder)

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


def test_video_source_exhaustion_stops_loop() -> None:
    total_frames = 5
    states = make_normal_states(total_frames)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder)

    supervisor.run_forever()

    video_source.release.assert_called_once()
    assert supervisor.processed_frame_count == total_frames


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
