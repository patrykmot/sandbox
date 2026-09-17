"""Fan-out and isolation for the alarm sinks.

The Supervisor takes one IAlarmHandler and calls it from inside the frame
loop's try block, so these two properties are what let a dashboard be an
alarm sink at all: every sink hears about every alarm, and a sink that throws
is the sink's problem, not the run's.
"""

from __future__ import annotations

import logging
from unittest.mock import create_autospec

import numpy as np
import pytest

from src.implementations.composite_alarm import CompositeAlarmHandler
from src.interfaces.alarm import AlarmEvent, IAlarmHandler
from src.interfaces.encoder import FeatureVector


def make_event(timestamp: int = 1_700_000_000_000) -> AlarmEvent:
    return AlarmEvent(
        timestamp=timestamp,
        vector=[FeatureVector(t=timestamp, vector=np.zeros(6))],
        anomaly_score=-0.5,
        description="Test anomaly.",
    )


def test_every_sink_receives_the_same_event() -> None:
    sinks = [create_autospec(IAlarmHandler, instance=True) for _ in range(3)]
    event = make_event()

    CompositeAlarmHandler(sinks).trigger_alarm(event)

    for sink in sinks:
        sink.trigger_alarm.assert_called_once_with(event)


def test_a_failing_sink_does_not_stop_the_others() -> None:
    first = create_autospec(IAlarmHandler, instance=True)
    exploding = create_autospec(IAlarmHandler, instance=True)
    exploding.trigger_alarm.side_effect = RuntimeError("dashboard exploded")
    last = create_autospec(IAlarmHandler, instance=True)

    CompositeAlarmHandler([first, exploding, last]).trigger_alarm(make_event())

    first.trigger_alarm.assert_called_once()
    last.trigger_alarm.assert_called_once()


def test_a_failing_sink_is_logged_not_swallowed_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence here would hide a permanently broken UI."""
    exploding = create_autospec(IAlarmHandler, instance=True)
    exploding.trigger_alarm.side_effect = RuntimeError("dashboard exploded")

    with caplog.at_level(logging.ERROR):
        CompositeAlarmHandler([exploding]).trigger_alarm(make_event())

    assert "dashboard exploded" in caplog.text


def test_no_sinks_is_not_an_error() -> None:
    """A headless run wires no sinks at all; raising an alarm must still work."""
    CompositeAlarmHandler([]).trigger_alarm(make_event())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
