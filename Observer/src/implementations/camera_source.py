"""Implementation 1 of IVideoSource: live camera / RTSP stream via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from src.interfaces.video_source import IVideoSource


class CameraVideoSource(IVideoSource):
    """Captures a live stream from a camera or RTSP URL using cv2.VideoCapture.

    Swap this out for a different IVideoSource implementation (e.g. a video
    file reader or a synthetic test source) without touching any other
    component - that's the point of the interface.
    """

    def __init__(self, camera_index: int | str = 0) -> None:
        """Args:
        camera_index: OpenCV device index (int) or an RTSP/video URL (str).
        """
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {camera_index!r}")

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        success, frame = self._capture.read()
        if not success:
            return False, None
        return True, frame

    def release(self) -> None:
        self._capture.release()
