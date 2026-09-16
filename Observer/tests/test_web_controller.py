"""HTTP-level tests for the FastAPI WebController.

No camera and no running server are needed: FastAPI's TestClient drives the
ASGI app in-process, and frames are plain numpy arrays.

These are skipped automatically where fastapi isn't installed, so the rest of
the suite still runs; install requirements.txt to exercise them.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from src.implementations.web_controller import WebController  # noqa: E402
from src.interfaces.alarm import AlarmEvent  # noqa: E402
from src.interfaces.controller import SystemStatus  # noqa: E402
from src.interfaces.encoder import FeatureVector  # noqa: E402
from src.interfaces.state import SupervisorState  # noqa: E402

FRAME = np.full((32, 32, 3), 128, dtype=np.uint8)


def make_status(
    state: SupervisorState = SupervisorState.COLLECTING_DATA,
    collected: int = 25,
    target: int = 100,
    frames: int = 25,
    alarms: int = 0,
) -> SystemStatus:
    return SystemStatus(
        state=state,
        processed_frame_count=frames,
        collection_progress=collected / target,
        collected_count=collected,
        collection_target=target,
        total_alarms=alarms,
    )


def make_alarm_event(score: float = -0.65, timestamp: int = 1_700_000_000_000) -> AlarmEvent:
    return AlarmEvent(
        timestamp=timestamp,
        vector=[FeatureVector(t=timestamp, vector=np.zeros(6))],
        anomaly_score=score,
        description="Test anomaly.",
        frame=FRAME,
    )


@pytest.fixture()
def controller() -> WebController:
    return WebController(max_alarms=20)


@pytest.fixture()
def client(controller: WebController) -> TestClient:
    return TestClient(controller.app)


def test_index_serves_dashboard(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # The dashboard must wire up the three data sources it depends on.
    for endpoint in ("/video_feed", "/api/status", "/alerts"):
        assert endpoint in response.text


def test_status_before_first_frame_is_safe(client: TestClient) -> None:
    """The UI may poll before the Supervisor thread has produced anything."""
    payload = client.get("/api/status").json()
    assert payload["state"] == SupervisorState.INITIALIZING.name
    assert payload["processed_frame_count"] == 0
    assert payload["collection_progress_percent"] == 0.0


def test_status_reflects_published_snapshot(
    controller: WebController, client: TestClient
) -> None:
    controller.publish_status(make_status(collected=25, target=100, frames=25))

    payload = client.get("/api/status").json()
    assert payload["state"] == "COLLECTING_DATA"
    assert payload["processed_frame_count"] == 25
    assert payload["collected_count"] == 25
    assert payload["collection_target"] == 100
    assert payload["collection_progress_percent"] == 25.0


def test_alerts_empty_by_default(client: TestClient) -> None:
    assert client.get("/alerts").json() == {"alarms": []}


def test_alerts_returns_newest_first_with_frame_urls(
    controller: WebController, client: TestClient
) -> None:
    controller.publish_alarm(make_alarm_event(score=-0.10, timestamp=1_700_000_000_000))
    controller.publish_alarm(make_alarm_event(score=-0.90, timestamp=1_700_000_005_000))

    alarms = client.get("/alerts").json()["alarms"]
    assert len(alarms) == 2

    # Newest first - that's the order the panel renders.
    assert alarms[0]["timestamp"] == 1_700_000_005_000
    assert alarms[0]["anomaly_score"] == -0.9
    assert alarms[0]["object_count"] == 1
    assert alarms[0]["frame_url"] == f"/api/alarms/{alarms[0]['id']}/frame"


def test_alarm_panel_keeps_only_last_n(client: TestClient) -> None:
    controller = WebController(max_alarms=20)
    client = TestClient(controller.app)

    for i in range(25):
        controller.publish_alarm(make_alarm_event(timestamp=1_700_000_000_000 + i))

    alarms = client.get("/alerts").json()["alarms"]
    assert len(alarms) == 20
    # The five oldest were evicted.
    assert alarms[-1]["timestamp"] == 1_700_000_000_005


def test_alarm_frame_endpoint_returns_jpeg(
    controller: WebController, client: TestClient
) -> None:
    controller.publish_alarm(make_alarm_event())
    alarm_id = client.get("/alerts").json()["alarms"][0]["id"]

    response = client.get(f"/api/alarms/{alarm_id}/frame")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")  # JPEG SOI marker


def test_unknown_alarm_frame_returns_404(client: TestClient) -> None:
    assert client.get("/api/alarms/9999/frame").status_code == 404


class FakeRequest:
    """Stands in for a Starlette Request, reporting a disconnect after N polls."""

    def __init__(self, disconnect_after: int) -> None:
        self._polls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._polls += 1
        return self._polls > self._disconnect_after


def test_video_feed_route_is_registered(controller: WebController) -> None:
    assert "/video_feed" in {route.path for route in controller.app.routes}


def test_mjpeg_stream_frames_and_stops_on_disconnect(controller: WebController) -> None:
    """Exercised at generator level rather than over HTTP.

    The MJPEG response is endless by design, so an HTTP-level test can only
    read a chunk and abandon the connection - it can never assert the loop
    *ends*. Terminating on client disconnect is the whole point (otherwise
    every closed browser tab leaks a spinning task), so that's asserted here
    directly.
    """
    controller.publish_frame(FRAME)
    request = FakeRequest(disconnect_after=2)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in controller._mjpeg_stream(request)]

    chunks = asyncio.run(collect())

    # Two frames yielded, then the disconnect ended the loop rather than
    # streaming forever.
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.startswith(b"--frame\r\n")
        assert b"Content-Type: image/jpeg" in chunk
        assert b"\xff\xd8" in chunk  # JPEG SOI marker in the payload


def test_mjpeg_stream_waits_without_yielding_before_first_frame(
    controller: WebController,
) -> None:
    """No frame published yet must not emit an empty/corrupt JPEG part."""
    request = FakeRequest(disconnect_after=2)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in controller._mjpeg_stream(request)]

    assert asyncio.run(collect()) == []