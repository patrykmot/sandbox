"""Interface 1: Video Data Source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraOption:
    """One selectable video source, as offered to the operator.

    Attributes:
        id: What gets handed to the IVideoSource factory - an OpenCV device
            index, or a stream/file URL.
        name: Human-readable label, e.g. "Logitech C920 HD Pro".
    """

    id: int | str
    name: str


class IVideoSource(ABC):
    """Abstract interface to stream frames from a hardware or software source.

    Concrete implementations (e.g. a live camera, an RTSP stream, a video file,
    or a synthetic/test source) plug in here without the rest of the system
    knowing the difference.
    """

    @abstractmethod
    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        """Retrieve the next available frame.

        Returns:
            A tuple of (success, frame). `success` is False when no frame
            could be read (e.g. stream ended or disconnected), in which case
            `frame` is None.
        """
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Release any underlying hardware/software resources."""
        raise NotImplementedError
