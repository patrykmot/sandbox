"""Interface 2 & 6: Video Encoder and Feature Encoder.

Data flow: Frame -> [IVideoEncoder] -> EncodedFrame(list[StateVector], annotated)
-> [IFeatureEncoder] -> list[FeatureVector].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StateVector:
    """A single detected object's kinematic state at a point in time.

    Attributes:
        t: Timestamp in milliseconds (Unix epoch, UTC) when this state was observed.
        x: 2D center X position (pixels).
        y: 2D center Y position (pixels).
        vx: Speed on X axis (pixels change / second).
        vy: Speed on Y axis (pixels change / second).
        size: Bounding box surface area (pixels^2).
        object_type: Class ID of the detected object (e.g. 0 for Person).
    """

    t: int
    x: float
    y: float
    vx: float
    vy: float
    size: float
    object_type: int


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    """Everything a single `IVideoEncoder.encode()` call produced.

    Both fields come from the SAME inference pass, which is why they are
    returned together rather than fetched separately: a later getter could
    hand back an annotated frame belonging to a different call once the
    Supervisor loop runs on its own thread.

    Attributes:
        states: One StateVector per detected/tracked object.
        annotated_frame: The frame with detection overlays drawn on it, for
            display purposes only (never fed to the ML pipeline).
            Implementations with nothing to draw return the input frame
            unchanged.
    """

    states: list[StateVector]
    annotated_frame: np.ndarray


class IVideoEncoder(ABC):
    """Abstract interface to extract objects from a frame as StateVectors."""

    @abstractmethod
    def encode(self, frame: np.ndarray) -> EncodedFrame:
        """Process a single frame into StateVectors plus a display overlay."""
        raise NotImplementedError

    def reset(self) -> None:
        """Forget per-object history (track ids, previous positions).

        Called when a scan starts, so velocities are never derived from
        tracks belonging to an earlier run or a different camera. Not
        abstract - a stateless encoder can ignore it.
        """
        return None


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """A fixed-size numeric feature representation derived from StateVector(s).

    Attributes:
        t: Timestamp in milliseconds for which this feature was computed
            (same as the originating StateVector.t).
        vector: N-sized array of float features. This is the exact array fed
            to IAnomalyDetector.fit()/predict() - it must always be the same
            length for a given IFeatureEncoder implementation.
    """

    t: int
    vector: np.ndarray


class IFeatureEncoder(ABC):
    """Abstract interface mapping list[StateVector] -> list[FeatureVector].

    Implementations decide how to turn raw kinematic states (potentially for
    multiple simultaneous objects) into the feature vectors used for anomaly
    detection. The output list size does not need to match the input list
    size.
    """

    @abstractmethod
    def encode(self, states: list[StateVector]) -> list[FeatureVector]:
        """Process StateVectors and produce FeatureVectors for the ML model."""
        raise NotImplementedError
