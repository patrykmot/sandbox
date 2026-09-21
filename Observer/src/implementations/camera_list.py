"""Enumerating the machine's cameras, for the dashboard's camera picker.

OpenCV has no enumeration API, so the only portable way to find out what is
there is to try opening device indices and keep the ones that answer. That
behaves the same on Windows, Linux and macOS and needs nothing beyond OpenCV
itself, at the cost of generic labels ("Camera 0") rather than product names.

Two details make that practical rather than merely correct:

  * **Probing is noisy.** Asking OpenCV for an index that does not exist
    makes its backends complain on stderr ("Camera index out of range",
    "VIDEOIO/FFMPEG: ... libavdevice"). Those messages are expected - they
    are how we learn the index is empty - so the probe runs with OpenCV's
    logging silenced and puts what it found in the application log instead.

  * **Probing is slow and repeated.** Each empty index costs the backend
    chain a moment, and the dashboard asks for the list on every page load
    and whenever it returns to IDLE - several times over with more than one
    browser tab open. Results are therefore cached briefly, so a burst of
    requests costs one probe.

The list is only ever an offer to the operator: opening the chosen source is
what really decides whether it works.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2

from src.interfaces.video_source import CameraOption

logger = logging.getLogger(__name__)

#: How long a probe result stays good. Long enough to absorb a page load's
#: worth of requests, short enough that a camera plugged in while the system
#: sits in IDLE shows up on the next refresh.
CACHE_TTL_SECONDS = 10.0

_cache_lock = threading.Lock()
_cache: tuple[float, int, list[CameraOption]] | None = None  # (expires_at, max_probe, options)

# TODO: check if there is simpler way to get cameras list.
def list_cameras(
    max_probe: int = 5,
    always_include: int | str | None = None,
    force_refresh: bool = False,
) -> list[CameraOption]:
    """Return the selectable cameras, best-effort and never raising.

    Args:
        max_probe: How many device indices to try (0..max_probe-1).
        always_include: A source that must appear in the list even if probing
            did not find it - typically the one already in use, which many
            backends refuse to open a second time and which would otherwise
            vanish from the dropdown while it is running.
        force_refresh: Ignore the cached result and probe again.
    """
    options = list(_probe_cached(max_probe, force_refresh))

    if always_include is not None and not any(o.id == always_include for o in options):
        options.append(CameraOption(id=always_include, name=label_for(always_include)))

    return options


def label_for(camera: int | str) -> str:
    """What to call a source in the UI: an index gets a number, a URL is itself."""
    return f"Camera {camera}" if isinstance(camera, int) else str(camera)


# --- Internals ------------------------------------------------------------------------


def _probe_cached(max_probe: int, force_refresh: bool) -> list[CameraOption]:
    global _cache

    with _cache_lock:
        cached = _cache
        fresh = (
            cached is not None
            and not force_refresh
            and cached[1] == max_probe
            and cached[0] > time.monotonic()
        )
        if fresh:
            assert cached is not None
            return cached[2]

        options = _probe_indices(max_probe)
        _cache = (time.monotonic() + CACHE_TTL_SECONDS, max_probe, options)

    logger.info(
        "Camera probe (indices 0-%d): found %s",
        max_probe - 1,
        ", ".join(o.name for o in options) or "nothing",
    )
    return options


def _probe_indices(max_probe: int) -> list[CameraOption]:
    """Open each index briefly to see whether a camera answers."""
    found: list[CameraOption] = []
    with _opencv_quiet():
        for index in range(max_probe):
            capture = None
            try:
                capture = cv2.VideoCapture(index)
                if capture.isOpened():
                    found.append(CameraOption(id=index, name=label_for(index)))
            except Exception:
                logger.debug("Probing camera index %d failed.", index, exc_info=True)
            finally:
                if capture is not None:
                    capture.release()
    return found


class _opencv_quiet:
    """Silence OpenCV's own logging for the duration of the probe.

    An absent device is normal here, but OpenCV reports it at ERROR level
    straight to stderr, where it reads like a fault in the application. Not
    every build exposes the logging controls, so failure to silence is
    ignored rather than fatal.
    """

    def __init__(self) -> None:
        self._previous: int | None = None

    def __enter__(self) -> "_opencv_quiet":
        try:
            self._previous = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:
            self._previous = None
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._previous is None:
            return
        try:
            cv2.utils.logging.setLogLevel(self._previous)
        except Exception:
            logger.debug("Restoring the OpenCV log level failed.", exc_info=True)
