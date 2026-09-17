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
from src.interfaces.controller import IController
from src.interfaces.encoder import EncodedFrame, IVideoEncoder, StateVector
from src.interfaces.video_source import IVideoSource

COLLECTION_TARGET = 60
BLANK_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)
ANNOTATED_FRAME = np.full((4, 4, 3), 255, dtype=np.uint8)


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

    The annotated frame is a distinct array from BLANK_FRAME so tests can
    assert the Supervisor forwards the *annotated* image, not the raw one.
    """
    mock_encoder = create_autospec(IVideoEncoder, instance=True)
    mock_encoder.encode.side_effect = [
        EncodedFrame(states=states, annotated_frame=ANNOTATED_FRAME)
        for states in state_sequence
    ]
    return mock_encoder


def build_supervisor(
    video_source: IVideoSource,
    video_encoder: IVideoEncoder,
    controller: IController | None = None,
) -> Supervisor:
    """A Supervisor already past IDLE, wired to one fixed mock source.

    The real system builds a source per camera selection; a test only ever
    wants the one mock, so the factory ignores the camera id.
    """
    supervisor = Supervisor(
        video_source_factory=lambda camera: video_source,
        video_encoder=video_encoder,
        feature_encoder=DummyFeatureEncoder(),
        anomaly_detector=IsolationForestAnomalyDetector(contamination=0.05),
        alarm_handler=ConsoleLoggerAlarmHandler(),
        collection_target_value=COLLECTION_TARGET,
        controller=controller,
    )
    # Tests drive step() directly, which is only meaningful once scanning.
    supervisor.initialize()
    supervisor.request_start()
    supervisor._drain_commands()  # apply it here rather than in run_forever()
    return supervisor


def test_reaches_monitoring_after_collection_target() -> None:
    # +1 frame after the collection target so we can observe MONITORING.
    total_frames = COLLECTION_TARGET + 1
    states = make_normal_states(total_frames)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder)

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

    for _ in range(outlier_index):
        assert supervisor.step() is True
    assert supervisor.state == SupervisorState.MONITORING

    # This step processes the injected outlier frame.
    assert supervisor.step() is True
    assert supervisor.state == SupervisorState.ALARM_ACTIVE

    alarm_handler = supervisor._alarm_handler  # test-only introspection
    assert len(alarm_handler.recent_alarms) == 1
    assert alarm_handler.recent_alarms[0].anomaly_score < 0


def test_collection_progress_is_printed_as_single_updating_line(capsys) -> None:
    total_frames = COLLECTION_TARGET + 1  # +1 to also observe a MONITORING-phase frame
    states = make_normal_states(total_frames)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder)

    for _ in range(COLLECTION_TARGET):
        assert supervisor.step() is True
    assert supervisor.state == SupervisorState.MONITORING

    out = capsys.readouterr().out

    # Every collection update is written on its own "\r"-prefixed segment
    # (no bare "\n" between them), so the whole collection phase prints as
    # one continuously-overwritten console line.
    segments = out.split("\r")
    progress_segments = [s for s in segments if "Collecting baseline data:" in s]
    assert len(progress_segments) == COLLECTION_TARGET
    assert all("\n" not in s for s in progress_segments[:-1])

    # The final update closes the line with a newline and reports 100%.
    assert progress_segments[-1].endswith("\n")
    assert f"({COLLECTION_TARGET}/{COLLECTION_TARGET})" in progress_segments[-1]
    assert "100.0%" in progress_segments[-1]

    # A monitoring-phase frame afterwards must not print another progress line.
    assert supervisor.step() is True
    out_after = capsys.readouterr().out
    assert "Collecting baseline data:" not in out_after


def test_video_source_exhaustion_enters_error_and_releases_the_camera() -> None:
    """A camera that stops delivering is an error the operator must see -
    the loop parks in ERROR rather than dying quietly."""
    total_frames = 5
    states = make_normal_states(total_frames)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder)

    for _ in range(total_frames):
        assert supervisor.step() is True

    assert supervisor.step() is False
    assert supervisor.state == SupervisorState.ERROR
    assert supervisor.error is not None
    video_source.release.assert_called_once()
    assert supervisor.processed_frame_count == total_frames


def test_stop_returns_to_idle_from_error() -> None:
    """STOP is the only button offered in ERROR, so it has to work there."""
    states = make_normal_states(1)
    video_source = build_mock_video_source(0)  # no frames at all
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder)

    assert supervisor.step() is False
    assert supervisor.state == SupervisorState.ERROR

    supervisor.request_stop()
    supervisor._drain_commands()

    assert supervisor.state == SupervisorState.IDLE
    assert supervisor.error is None


def test_supervisor_pushes_annotated_frame_and_status_to_controller() -> None:
    total_frames = 3
    states = make_normal_states(total_frames)

    controller = create_autospec(IController, instance=True)
    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder, controller=controller)

    for _ in range(total_frames):
        assert supervisor.step() is True

    assert controller.publish_frame.call_count == total_frames
    assert controller.publish_status.call_count == total_frames

    # The UI must receive the annotated frame, never the raw camera frame.
    pushed_frame = controller.publish_frame.call_args[0][0]
    assert np.array_equal(pushed_frame, ANNOTATED_FRAME)

    status = controller.publish_status.call_args[0][0]
    assert status.state == SupervisorState.COLLECTING_DATA
    assert status.processed_frame_count == total_frames
    assert status.collected_count == total_frames
    assert status.collection_target == COLLECTION_TARGET
    assert status.total_alarms == 0


def test_supervisor_pushes_alarm_to_controller() -> None:
    outlier_index = COLLECTION_TARGET + 3
    total_frames = outlier_index + 1

    states = make_normal_states(total_frames)
    states[outlier_index] = make_outlier_state(states[outlier_index][0].t)

    controller = create_autospec(IController, instance=True)
    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder, controller=controller)

    for _ in range(total_frames):
        assert supervisor.step() is True

    assert supervisor.state == SupervisorState.ALARM_ACTIVE
    controller.publish_alarm.assert_called_once()

    event = controller.publish_alarm.call_args[0][0]
    assert event.anomaly_score < 0
    assert np.array_equal(event.frame, ANNOTATED_FRAME)
    assert supervisor.total_alarms == 1


def test_controller_failure_does_not_break_processing_loop() -> None:
    """A misbehaving UI must never take down the processing loop."""
    total_frames = 3
    states = make_normal_states(total_frames)

    controller = create_autospec(IController, instance=True)
    controller.publish_status.side_effect = RuntimeError("UI exploded")
    controller.publish_frame.side_effect = RuntimeError("UI exploded")

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder, controller=controller)

    for _ in range(total_frames):
        assert supervisor.step() is True

    assert supervisor.state == SupervisorState.COLLECTING_DATA
    assert supervisor.processed_frame_count == total_frames


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
