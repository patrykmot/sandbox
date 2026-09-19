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
from src.implementations.composite_alarm import CompositeAlarmHandler
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector
from src.interfaces.alarm import IAlarmHandler
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
    alarm_handler: IAlarmHandler | None = None,
) -> Supervisor:
    """A Supervisor already past IDLE, wired to one fixed mock source.

    The real system builds a source per camera selection; a test only ever
    wants the one mock, so the factory ignores the camera id.
    """
    supervisor = Supervisor(
        video_source_factory=lambda camera: video_source,
        video_encoder=video_encoder,
        feature_encoder=DummyFeatureEncoder(),
        anomaly_detector_factory=lambda name: IsolationForestAnomalyDetector(
            contamination=0.05
        ),
        alarm_handler=alarm_handler or ConsoleLoggerAlarmHandler(),
        collection_target_value=COLLECTION_TARGET,
        detector="isolation_forest",
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


def test_latest_frame_and_status_expose_what_a_ui_needs() -> None:
    """Nothing is pushed anywhere: a UI pulls these two, so they must be
    right after every frame."""
    total_frames = 3
    states = make_normal_states(total_frames)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder)

    for _ in range(total_frames):
        assert supervisor.step() is True

    # The UI must see the annotated frame, never the raw camera frame.
    assert np.array_equal(supervisor.latest_frame, ANNOTATED_FRAME)

    status = supervisor.build_status()
    assert status.state == SupervisorState.COLLECTING_DATA
    assert status.processed_frame_count == total_frames
    assert status.collected_count == total_frames
    assert status.collection_target == COLLECTION_TARGET
    assert status.total_alarms == 0


def test_alarm_reaches_the_alarm_handler() -> None:
    """The handler is the one and only way an event leaves the Supervisor."""
    outlier_index = COLLECTION_TARGET + 3
    total_frames = outlier_index + 1

    states = make_normal_states(total_frames)
    states[outlier_index] = make_outlier_state(states[outlier_index][0].t)

    alarm_handler = create_autospec(IAlarmHandler, instance=True)
    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(video_source, video_encoder, alarm_handler=alarm_handler)

    for _ in range(total_frames):
        assert supervisor.step() is True

    assert supervisor.state == SupervisorState.ALARM_ACTIVE
    alarm_handler.trigger_alarm.assert_called_once()

    event = alarm_handler.trigger_alarm.call_args[0][0]
    assert event.anomaly_score < 0
    assert np.array_equal(event.frame, ANNOTATED_FRAME)
    assert supervisor.total_alarms == 1


def test_a_failing_alarm_sink_does_not_break_processing_loop() -> None:
    """A misbehaving UI must never take down the processing loop.

    The UI is now an alarm sink like any other, and the Supervisor calls the
    handler from inside the frame loop's try block - so the isolation has to
    come from the composite. This is that guarantee, end to end.
    """
    outlier_index = COLLECTION_TARGET + 3
    total_frames = outlier_index + 2

    states = make_normal_states(total_frames)
    states[outlier_index] = make_outlier_state(states[outlier_index][0].t)

    exploding = create_autospec(IAlarmHandler, instance=True)
    exploding.trigger_alarm.side_effect = RuntimeError("UI exploded")
    surviving = create_autospec(IAlarmHandler, instance=True)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = build_supervisor(
        video_source,
        video_encoder,
        alarm_handler=CompositeAlarmHandler([exploding, surviving]),
    )

    for _ in range(total_frames):
        assert supervisor.step() is True

    # However many alarms this run raised, the sink behind the failing one
    # heard every single one of them.
    assert exploding.trigger_alarm.call_count >= 1
    assert surviving.trigger_alarm.call_count == exploding.trigger_alarm.call_count
    assert supervisor.state != SupervisorState.ERROR
    assert supervisor.processed_frame_count == total_frames


def test_training_progress_is_reported_while_fitting() -> None:
    """The dashboard's training bar is driven by whatever fit() reports, so
    the Supervisor has to hand the detector a callback and keep the result."""
    total_frames = COLLECTION_TARGET + 1
    states = make_normal_states(total_frames)

    seen: list[float] = []

    class RecordingDetector(IsolationForestAnomalyDetector):
        def fit(self, training_data, progress_callback=None):
            super().fit(training_data)
            for percent in (25, 50, 75):
                progress_callback(percent)
                # What a UI polling build_status() would read at this instant.
                seen.append(supervisor.build_status().training_progress)

    video_source = build_mock_video_source(total_frames)
    video_encoder = build_mock_video_encoder(states)
    supervisor = Supervisor(
        video_source_factory=lambda camera: video_source,
        video_encoder=video_encoder,
        feature_encoder=DummyFeatureEncoder(),
        anomaly_detector_factory=lambda name: RecordingDetector(contamination=0.05),
        alarm_handler=ConsoleLoggerAlarmHandler(),
        collection_target_value=COLLECTION_TARGET,
    )
    supervisor.initialize()
    supervisor.request_start()
    supervisor._drain_commands()

    for _ in range(total_frames):
        assert supervisor.step() is True

    assert seen == [0.25, 0.50, 0.75]
    assert supervisor.state == SupervisorState.MONITORING


def test_a_fresh_detector_is_built_for_every_run() -> None:
    """A run must never inherit the previous run's fitted weights."""
    built: list[str] = []

    def factory(name: str) -> IsolationForestAnomalyDetector:
        built.append(name)
        return IsolationForestAnomalyDetector(contamination=0.05)

    supervisor = Supervisor(
        video_source_factory=lambda camera: build_mock_video_source(1),
        video_encoder=build_mock_video_encoder(make_normal_states(1)),
        feature_encoder=DummyFeatureEncoder(),
        anomaly_detector_factory=factory,
        alarm_handler=ConsoleLoggerAlarmHandler(),
        detector="isolation_forest",
    )
    supervisor.initialize()

    for _ in range(2):
        supervisor.request_start()
        supervisor._drain_commands()
        supervisor.request_stop()
        supervisor._drain_commands()

    assert built == ["isolation_forest", "isolation_forest"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
