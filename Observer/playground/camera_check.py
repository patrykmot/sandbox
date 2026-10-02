"""Quick RTSP camera check: connect, show the live view in a window, log every error.

Scratch tool, not part of the running system. Edit the values below and run:

    python playground/camera_check.py

Press Q or Esc (or close the window) to quit. Everything is logged to the console.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from urllib.parse import quote

# --- Camera settings - edit these -------------------------------------------------
HOST = "192.168.1.181"
PORT = 554
USERNAME = "XXX"
PASSWORD = "XXX"
STREAM_PATH = "stream1"  # e.g. "Streaming/Channels/101" (Hikvision), "cam/realmonitor?channel=1&subtype=0" (Dahua)
USE_TCP = True  # TCP is slower to start but far more reliable than UDP over Wi-Fi/VPN

OPEN_TIMEOUT_MS = 10_000  # give up connecting after this long
READ_TIMEOUT_MS = 5_000  # treat the stream as dead after this long without a frame
MAX_FAILED_READS = 30  # consecutive failed reads before trying to reconnect
RECONNECT_DELAY_S = 3.0
MAX_RECONNECTS = 5  # 0 = never reconnect, just quit on the first dropout

WINDOW_NAME = "RTSP camera check"

# Must be set BEFORE cv2 is imported/opens the stream, or FFmpeg ignores it.
if USE_TCP:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2  # noqa: E402  (import after the env var on purpose)

logger = logging.getLogger("camera_check")


def build_url(with_password: bool = True) -> str:
    """rtsp://user:pass@host:port/path - credentials are URL-escaped so @, : or / in them don't break the URL."""
    password = quote(PASSWORD, safe="") if with_password else "****"
    auth = f"{quote(USERNAME, safe='')}:{password}@" if USERNAME else ""
    return f"rtsp://{auth}{HOST}:{PORT}/{STREAM_PATH.lstrip('/')}"


def open_stream() -> cv2.VideoCapture | None:
    """Open the RTSP stream, or return None (and log why) if it can't be opened."""
    logger.info("Connecting to %s (transport: %s) ...", build_url(with_password=False), "TCP" if USE_TCP else "UDP")
    try:
        cap = cv2.VideoCapture(
            build_url(),
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS],
        )
    except cv2.error as exc:
        logger.error("OpenCV error while opening the stream: %s", exc)
        return None

    if not cap.isOpened():
        logger.error(
            "Could not open the stream. Check host/port reachability, credentials, stream path, "
            "and that the camera allows RTSP."
        )
        cap.release()
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("Connected: %dx%d @ %.1f fps (as reported by the camera).", width, height, fps)
    return cap


def reconnect(used: int) -> tuple[cv2.VideoCapture | None, int]:
    """Try to reopen the stream until MAX_RECONNECTS attempts in total are used.

    Returns the new capture (None when every attempt failed) and the updated
    count of attempts used, so the budget covers the whole session.
    """
    while used < MAX_RECONNECTS:
        used += 1
        logger.warning("Stream lost. Reconnecting in %.0f s (attempt %d/%d) ...", RECONNECT_DELAY_S, used, MAX_RECONNECTS)
        time.sleep(RECONNECT_DELAY_S)
        cap = open_stream()
        if cap is not None:
            return cap, used
    logger.error("Stream lost and all %d reconnect attempt(s) used up - giving up.", MAX_RECONNECTS)
    return None, used


def window_closed() -> bool:
    """True once the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def run() -> int:
    """Main loop. Returns the process exit code."""
    cap = open_stream()
    if cap is None:
        return 1

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        # Typically opencv-python-headless, which has no GUI support.
        logger.error("Cannot create a window (is opencv-python-headless installed instead of opencv-python?): %s", exc)
        cap.release()
        return 1

    failed_reads = 0
    reconnects = 0
    frames = 0
    started = time.monotonic()

    try:
        while True:
            try:
                ok, frame = cap.read()
            except cv2.error as exc:
                logger.error("OpenCV error while reading a frame: %s", exc)
                ok, frame = False, None

            if not ok or frame is None:
                failed_reads += 1
                logger.warning("Failed to read frame (%d/%d in a row).", failed_reads, MAX_FAILED_READS)
                if failed_reads < MAX_FAILED_READS:
                    continue

                # The stream is considered dead - try to reconnect.
                cap.release()
                cap, reconnects = reconnect(used=reconnects)
                if cap is None:
                    return 1
                failed_reads = 0
                continue

            if failed_reads:
                logger.info("Stream recovered after %d failed read(s).", failed_reads)
            failed_reads = 0
            frames += 1

            try:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as exc:
                logger.error("OpenCV error while displaying a frame: %s", exc)
                return 1

            if key in (ord("q"), ord("Q"), 27):  # 27 = Esc
                logger.info("Quit requested.")
                return 0
            if window_closed():
                logger.info("Window closed.")
                return 0

    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C).")
        return 0
    except Exception:  # anything unexpected: log the full traceback, don't just die silently
        logger.exception("Unexpected error.")
        return 1
    finally:
        elapsed = time.monotonic() - started
        if frames:
            logger.info("Shown %d frames in %.1f s (%.1f fps).", frames, elapsed, frames / elapsed if elapsed else 0.0)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(run())
