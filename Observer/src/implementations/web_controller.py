"""Implementation 1 of IController: a FastAPI web dashboard.

Threading model
---------------
The Supervisor loop runs on a background thread and *pushes* into this
object (publish_status / publish_frame / publish_alarm). FastAPI serves
requests on the main thread. Every piece of shared mutable state below is
therefore guarded by a single `threading.Lock`.

Frames are JPEG-encoded once, at push time, rather than per connected
client - the MJPEG stream then just re-sends the stored bytes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.interfaces.alarm import AlarmEvent
from src.interfaces.controller import IController, IControlTarget, SystemStatus
from src.interfaces.state import SupervisorState
from src.interfaces.video_source import CameraOption

logger = logging.getLogger(__name__)

_DASHBOARD_FILE = Path(__file__).with_name("dashboard.html")
#: Locally served CSS/JS (Bootstrap, jQuery). Vendored on purpose: the
#: dashboard must load on a machine that can reach this server but has no
#: internet access, so nothing may come from a CDN.
_STATIC_DIR = Path(__file__).with_name("static")
_MJPEG_BOUNDARY = "frame"


@dataclass(frozen=True, slots=True)
class AlarmRecord:
    """One row of the alarm panel, with its Camera Frame View already encoded."""

    id: int
    timestamp: int
    anomaly_score: float
    description: str
    object_count: int
    frame_jpeg: bytes | None

    @property
    def iso_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp / 1000.0, tz=timezone.utc).isoformat(
            timespec="seconds"
        )


class WebController(IController):
    """Serves the live dashboard, MJPEG stream, status and alarm APIs."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_alarms: int = 20,
        stream_fps: int = 15,
        jpeg_quality: int = 80,
        control: IControlTarget | None = None,
        cameras_provider: Callable[[int | str | None], list[CameraOption]] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        # Commands go out through this; without one the dashboard is
        # read-only and the control endpoints report 503.
        self._control = control
        self._cameras_provider = cameras_provider
        self._stream_interval = 1.0 / max(1, stream_fps)
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._status: SystemStatus | None = None
        self._alarms: deque[AlarmRecord] = deque(maxlen=max_alarms)
        self._next_alarm_id = 1

        self.app = self._build_app()

    def bind_control(self, control: IControlTarget) -> None:
        """Attach the Supervisor after construction.

        The two know about each other in opposite directions - the Supervisor
        pushes into this controller, this controller requests from the
        Supervisor - so one of the two links has to be made after both exist.
        """
        self._control = control

    # --- IController (called from the Supervisor's background thread) -----------------

    def publish_status(self, status: SystemStatus) -> None:
        with self._lock:
            self._status = status

    def publish_frame(self, annotated_frame: np.ndarray) -> None:
        jpeg = self._encode_jpeg(annotated_frame)
        if jpeg is None:
            return
        with self._lock:
            self._latest_jpeg = jpeg

    def publish_alarm(self, event: AlarmEvent) -> None:
        frame_jpeg = self._encode_jpeg(event.frame) if event.frame is not None else None
        with self._lock:
            record = AlarmRecord(
                id=self._next_alarm_id,
                timestamp=event.timestamp,
                anomaly_score=event.anomaly_score,
                description=event.description,
                object_count=len(event.vector),
                frame_jpeg=frame_jpeg,
            )
            self._next_alarm_id += 1
            self._alarms.append(record)

    def clear_alarms(self) -> None:
        """Drop the alarm panel's contents (a new scan has started)."""
        with self._lock:
            self._alarms.clear()

    def run(self) -> None:
        """Block serving the dashboard (main thread)."""
        uvicorn.run(self.app, host=self._host, port=self._port, log_level="info")

    # --- Helpers ----------------------------------------------------------------------

    def _encode_jpeg(self, frame: np.ndarray) -> bytes | None:
        success, buffer = cv2.imencode(".jpg", frame, self._jpeg_params)
        if not success:
            return None
        return buffer.tobytes()

    def _require_control(self) -> IControlTarget:
        if self._control is None:
            raise HTTPException(status_code=503, detail="This dashboard is read-only.")
        return self._control

    def _status_payload(self) -> dict:
        with self._lock:
            status = self._status
            total_alarms = len(self._alarms)

        if status is None:
            # The Supervisor thread hasn't produced its first frame yet.
            return {
                "state": SupervisorState.INITIALIZING.name,
                "processed_frame_count": 0,
                "collection_progress": 0.0,
                "collection_progress_percent": 0.0,
                "collected_count": 0,
                "collection_target": 0,
                "total_alarms": total_alarms,
                "camera": None,
                "error": None,
            }

        return {
            "state": status.state.name,
            "processed_frame_count": status.processed_frame_count,
            "collection_progress": status.collection_progress,
            "collection_progress_percent": round(status.collection_progress * 100, 1),
            "collected_count": status.collected_count,
            "collection_target": status.collection_target,
            "total_alarms": status.total_alarms,
            "camera": status.camera,
            "error": status.error,
        }

    async def _mjpeg_stream(self, request: Request):
        """Yield multipart JPEG chunks for as long as the client stays connected.

        Deliberately an *async* generator: a sync one would occupy a
        threadpool worker per viewer, and - because this loop is endless -
        would keep spinning after the browser tab closed. Polling
        `is_disconnected()` ends the loop when the client goes away.
        """
        while True:
            if await request.is_disconnected():
                break

            with self._lock:
                jpeg = self._latest_jpeg

            if jpeg is not None:
                yield (
                    f"--{_MJPEG_BOUNDARY}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode() + jpeg + b"\r\n"

            await asyncio.sleep(self._stream_interval)

    # --- Routes -----------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Observer Dashboard", docs_url=None, redoc_url=None)

        # Serves src/implementations/static/ at /static - every asset the
        # dashboard needs comes from this server, never from the internet.
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            return HTMLResponse(_DASHBOARD_FILE.read_text(encoding="utf-8"))

        @app.get("/api/status")
        def api_status() -> dict:
            return self._status_payload()

        @app.get("/video_feed")
        def video_feed(request: Request) -> StreamingResponse:
            return StreamingResponse(
                self._mjpeg_stream(request),
                media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
            )

        # --- Control -----------------------------------------------------------------
        #
        # Every one of these only *requests* something: the Supervisor applies
        # it on its own thread at a frame boundary, and the next /api/status
        # poll is what tells the browser whether it happened.

        @app.get("/api/cameras")
        def api_cameras() -> dict:
            with self._lock:
                selected = self._status.camera if self._status else None

            if self._cameras_provider is None:
                cameras: list[CameraOption] = []
            else:
                try:
                    cameras = self._cameras_provider(selected)
                except Exception:
                    logger.exception("Listing cameras failed; returning an empty list.")
                    cameras = []

            return {
                "cameras": [{"id": c.id, "name": c.name} for c in cameras],
                "selected": selected,
            }

        @app.post("/api/control/start")
        def api_start() -> dict:
            self._require_control().request_start()
            return {"requested": "start"}

        @app.post("/api/control/stop")
        def api_stop() -> dict:
            self._require_control().request_stop()
            return {"requested": "stop"}

        @app.post("/api/control/camera")
        def api_camera(payload: dict = Body(...)) -> dict:
            if "camera" not in payload:
                raise HTTPException(status_code=422, detail="Missing 'camera'.")
            camera = _coerce_camera(payload["camera"])
            self._require_control().request_camera(camera)
            return {"requested": "camera", "camera": camera}

        @app.get("/alerts")
        def alerts() -> dict:
            with self._lock:
                records = list(self._alarms)

            # Newest first - that's the order the panel displays them in.
            return {
                "alarms": [
                    {
                        "id": record.id,
                        "timestamp": record.timestamp,
                        "time": record.iso_time,
                        "anomaly_score": round(record.anomaly_score, 4),
                        "description": record.description,
                        "object_count": record.object_count,
                        "frame_url": (
                            f"/api/alarms/{record.id}/frame" if record.frame_jpeg else None
                        ),
                    }
                    for record in reversed(records)
                ]
            }

        @app.get("/api/alarms/{alarm_id}/frame")
        def alarm_frame(alarm_id: int) -> Response:
            with self._lock:
                match = next((r for r in self._alarms if r.id == alarm_id), None)

            if match is None or match.frame_jpeg is None:
                raise HTTPException(status_code=404, detail="Alarm frame not available")
            return Response(content=match.frame_jpeg, media_type="image/jpeg")

        return app


def _coerce_camera(value: object) -> int | str:
    """A camera id is an OpenCV index or a stream URL.

    JSON gives us whichever the browser sent, and a <select> always sends
    strings - so "1" has to become 1, while "rtsp://..." stays as it is.
    """
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail="Invalid camera id.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Invalid camera id.")
        try:
            return int(text)
        except ValueError:
            return text
    raise HTTPException(status_code=422, detail="Invalid camera id.")
