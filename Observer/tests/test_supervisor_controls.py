"""The START / STOP / camera-selection behaviour behind the dashboard's one button.

No camera is needed: IVideoSource is mocked, and commands are drained
explicitly instead of running the loop on a thread.
"""

from __future__ import annotations

from unittest.mock import create_autospec

import numpy as np
import pytest

from src.core.supervisor import Supervisor, SupervisorState
from src.implementations.console_alarm import ConsoleLoggerAlarmHandler
from src.implementations.dummy_feature_encoder import DummyFeatureEncoder
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector
from src.interfaces.encoder import EncodedFrame, IVideoEncoder, StateVector
from src.interfaces.video_source import IVideoSource

RAW_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)
ANNOTATED_FRAME = np.full((4, 4, 3), 255, dtype=np.uint8)


def make_source() -> IVideoSource:
    source = create_autospec(IVideoSource, instance=True)
    source.get_frame.return_value = (True, RAW_FRAME)
    return source


def make_encoder() -> IVideoEncoder:
    encoder = create_autospec(IVideoEncoder, instance=True)
    encoder.encode.return_value = EncodedFrame(
        states=[StateVector(t=1, x=1.0, y=1.0, vx=0.0, vy=0.0, size=10.0, object_type=0)],
        annotated_frame=ANNOTATED_FRAME,
    )
    return encoder


def build(
    opened: list[int | str] | None = None,
    source: IVideoSource | None = None,
    encoder: IVideoEncoder | None = None,
) -> Supervisor:
    source = source or make_source()

    def factory(camera: int | str) -> IVideoSource:
        if opened is not None:
            opened.append(camera)
        return source

    supervisor = Supervisor(
        video_source_factory=factory,
        video_encoder=encoder or make_encoder(),
        feature_encoder=DummyFeatureEncoder(),
        anomaly_detector=IsolationForestAnomalyDetector(contamination=0.05),
        alarm_handler=ConsoleLoggerAlarmHandler(),
        collection_target_value=10,
        camera=0,
    )
    supervisor.initialize()
    return supervisor


def pump(supervisor: Supervisor) -> None:
    """Apply queued commands the way run_forever() would."""
    supervisor._drain_commands()


def test_boots_into_idle_without_opening_anything() -> None:
    opened: list[int | str] = []
    supervisor = build(opened=opened)

    assert supervisor.state == SupervisorState.IDLE
    assert opened == []  # the camera opens on the first preview pass, not at construction


def test_idle_previews_the_raw_frame_without_detecting() -> None:
    """The preview is a viewfinder: no encoder, no features, no scoring."""
    encoder = make_encoder()
    supervisor = build(encoder=encoder)

    supervisor.tick()

    encoder.encode.assert_not_called()
    assert supervisor.processed_frame_count == 0
    assert np.array_equal(supervisor.latest_frame, RAW_FRAME)


def test_start_moves_to_collecting_and_begins_a_new_run() -> None:
    """The run id is how a UI knows the previous run's alarms are stale -
    nobody has to be told to forget them."""
    supervisor = build()
    before = supervisor.build_status().run_id

    supervisor.request_start()
    pump(supervisor)

    assert supervisor.state == SupervisorState.COLLECTING_DATA
    assert supervisor.build_status().run_id == before + 1


def test_stop_keeps_the_run_id() -> None:
    """STOP discards the baseline but keeps the panel: the alarms of the run
    that just ended are still the alarms worth looking at."""
    supervisor = build()
    supervisor.request_start()
    pump(supervisor)
    after_start = supervisor.build_status().run_id

    supervisor.request_stop()
    pump(supervisor)

    assert supervisor.build_status().run_id == after_start


def test_a_failed_start_does_not_begin_a_new_run() -> None:
    """A camera that will not open never scanned, so it never invalidated
    what the panel is showing."""

    def factory(camera: int | str) -> IVideoSource:
        raise RuntimeError("device busy")

    supervisor = Supervisor(
        video_source_factory=factory,
        video_encoder=make_encoder(),
        feature_encoder=DummyFeatureEncoder(),
        anomaly_detector=IsolationForestAnomalyDetector(),
        alarm_handler=ConsoleLoggerAlarmHandler(),
        camera=7,
    )
    supervisor.initialize()
    before = supervisor.build_status().run_id

    supervisor.request_start()
    pump(supervisor)

    assert supervisor.build_status().run_id == before


def test_start_resets_tracking_state() -> None:
    """Velocities must never be derived from a previous run's track ids."""
    encoder = make_encoder()
    supervisor = build(encoder=encoder)

    supervisor.request_start()
    pump(supervisor)

    encoder.reset.assert_called_once()


def test_scanning_uses_the_annotated_frame() -> None:
    supervisor = build()
    supervisor.request_start()
    pump(supervisor)

    supervisor.tick()

    assert supervisor.processed_frame_count == 1
    assert np.array_equal(supervisor.latest_frame, ANNOTATED_FRAME)


def test_stop_returns_to_idle_and_discards_progress() -> None:
    supervisor = build()
    supervisor.request_start()
    pump(supervisor)
    for _ in range(3):
        supervisor.tick()
    assert supervisor.processed_frame_count == 3

    supervisor.request_stop()
    pump(supervisor)

    assert supervisor.state == SupervisorState.IDLE
    assert supervisor.processed_frame_count == 0
    assert supervisor.collection_progress == 0.0


def test_camera_can_be_changed_while_idle() -> None:
    opened: list[int | str] = []
    supervisor = build(opened=opened)
    supervisor.tick()               # opens camera 0 for preview
    assert opened == [0]

    supervisor.request_camera(2)
    pump(supervisor)
    supervisor.tick()               # reopens on the new selection

    assert supervisor.camera == 2
    assert opened == [0, 2]


def test_camera_change_is_ignored_while_scanning() -> None:
    """The camera is part of what the model learned, so it is locked once
    scanning starts - the UI disables the dropdown for the same reason."""
    supervisor = build()
    supervisor.request_start()
    pump(supervisor)

    supervisor.request_camera(3)
    pump(supervisor)

    assert supervisor.camera == 0


def test_start_is_ignored_unless_idle() -> None:
    supervisor = build()
    supervisor.request_start()
    pump(supervisor)
    supervisor.tick()
    frames_before = supervisor.processed_frame_count

    supervisor.request_start()      # a second START must not restart collection
    pump(supervisor)

    assert supervisor.state == SupervisorState.COLLECTING_DATA
    assert supervisor.processed_frame_count == frames_before


def test_failing_camera_keeps_the_system_idle_with_an_error() -> None:
    """A camera that will not open is a bad choice, not a broken system:
    stay IDLE so another one can be picked."""

    def factory(camera: int | str) -> IVideoSource:
        raise RuntimeError("device busy")

    supervisor = Supervisor(
        video_source_factory=factory,
        video_encoder=make_encoder(),
        feature_encoder=DummyFeatureEncoder(),
        anomaly_detector=IsolationForestAnomalyDetector(),
        alarm_handler=ConsoleLoggerAlarmHandler(),
        camera=7,
    )
    supervisor.initialize()

    supervisor.request_start()
    pump(supervisor)

    assert supervisor.state == SupervisorState.IDLE
    assert "device busy" in (supervisor.error or "")


def test_shutdown_ends_run_forever() -> None:
    supervisor = build()
    supervisor.shutdown()

    supervisor.run_forever()  # returns instead of looping

    assert supervisor.state == SupervisorState.IDLE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
