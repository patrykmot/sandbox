"""Interface 5: Alarm Handler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.interfaces.encoder import FeatureVector


@dataclass(frozen=True, slots=True)
class AlarmEvent:
    """Payload describing a detected anomaly.

    Attributes:
        timestamp: Timestamp in milliseconds (Unix epoch, UTC) of the event.
        vector: The FeatureVector(s) that triggered the alarm.
        frame: Optional snapshot of the frame at the time of the alarm.
        anomaly_score: The raw anomaly score returned by the detector.
        description: Human-readable description of the event.
    """

    timestamp: int
    vector: list[FeatureVector]
    anomaly_score: float
    description: str
    frame: np.ndarray | None = None


class IAlarmHandler(ABC):
    """Abstract interface to dispatch notifications when an anomaly is detected."""

    @abstractmethod
    def trigger_alarm(self, event: AlarmEvent) -> None:
        """Handle the alarm payload (log it, send a notification, etc.)."""
        raise NotImplementedError
