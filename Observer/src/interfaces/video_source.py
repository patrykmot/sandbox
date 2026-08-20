"""Interface 1: Video Data Source."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


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
