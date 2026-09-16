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
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.interfaces.alarm import AlarmEvent
from src.interfaces.controller import IController, SystemStatus
from src.interfaces.state import SupervisorState

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
    ) -> None:
        self._host = host
        self._port = port
        self._stream_interval = 1.0 / max(1, stream_fps)
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._status: SystemStatus | None = None
        self._alarms: deque[AlarmRecord] = deque(maxlen=max_alarms)
        self._next_alarm_id = 1

        self.app = self._build_app()

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

    def run(self) -> None:
        """Block serving the dashboard (main thread)."""
        uvicorn.run(self.app, host=self._host, port=self._port, log_level="info")

    # --- Helpers ----------------------------------------------------------------------

    def _encode_jpeg(self, frame: np.ndarray) -> bytes | None:
        success, buffer = cv2.imencode(".jpg", frame, self._jpeg_params)
        if not success:
            return None
        return buffer.tobytes()

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
            }

        return {
            "state": status.state.name,
            "processed_frame_count": status.processed_frame_count,
            "collection_progress": status.collection_progress,
            "collection_progress_percent": round(status.collection_progress * 100, 1),
            "collected_count": status.collected_count,
            "collection_target": status.collection_target,
            "total_alarms": status.total_alarms,
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
