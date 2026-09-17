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
from src.interfaces.controller import ISupervisorPort, SystemStatus  # noqa: E402
from src.interfaces.encoder import FeatureVector  # noqa: E402
from src.interfaces.video_source import CameraOption  # noqa: E402
from src.interfaces.state import SupervisorState  # noqa: E402

FRAME = np.full((32, 32, 3), 128, dtype=np.uint8)


def make_status(
    state: SupervisorState = SupervisorState.COLLECTING_DATA,
    collected: int = 25,
    target: int = 100,
    frames: int = 25,
    alarms: int = 0,
    run_id: int = 1,
) -> SystemStatus:
    return SystemStatus(
        state=state,
        processed_frame_count=frames,
        collection_progress=collected / target,
        collected_count=collected,
        collection_target=target,
        total_alarms=alarms,
        run_id=run_id,
    )


def make_alarm_event(score: float = -0.65, timestamp: int = 1_700_000_000_000) -> AlarmEvent:
    return AlarmEvent(
        timestamp=timestamp,
        vector=[FeatureVector(t=timestamp, vector=np.zeros(6))],
        anomaly_score=score,
        description="Test anomaly.",
        frame=FRAME,
    )


class FakeSupervisor:
    """A stand-in for the real Supervisor: the ISupervisorPort surface, no loop.

    The dashboard pulls, so a test only has to decide what a pull returns.
    """

    def __init__(self, status: SystemStatus | None = None) -> None:
        self.status = status if status is not None else make_status()
        self.latest_frame: np.ndarray | None = None
        self.starts = 0
        self.stops = 0
        self.cameras: list[int | str] = []

    def build_status(self) -> SystemStatus:
        return self.status

    def request_start(self) -> None:
        self.starts += 1

    def request_stop(self) -> None:
        self.stops += 1

    def request_camera(self, camera: int | str) -> None:
        self.cameras.append(camera)


def test_fake_supervisor_satisfies_the_port() -> None:
    """If this drifts from the real port the HTTP tests below are fiction."""
    assert isinstance(FakeSupervisor(), ISupervisorPort)


@pytest.fixture()
def controller() -> WebController:
    return WebController(max_alarms=20)


@pytest.fixture()
def client(controller: WebController) -> TestClient:
    return TestClient(controller.app)


@pytest.fixture()
def supervisor() -> FakeSupervisor:
    return FakeSupervisor()


@pytest.fixture()
def bound(controller: WebController, supervisor: FakeSupervisor) -> WebController:
    """The same controller as `controller`, with a Supervisor attached."""
    controller.bind_supervisor(supervisor)
    return controller


def test_index_serves_dashboard(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # The dashboard must wire up the three data sources it depends on.
    for endpoint in ("/video_feed", "/api/status", "/alerts"):
        assert endpoint in response.text


def test_status_without_a_supervisor_is_safe(client: TestClient) -> None:
    """A dashboard with nothing bound is read-only, not broken - and the page
    may well poll before the wiring in main() has finished."""
    payload = client.get("/api/status").json()
    assert payload["state"] == SupervisorState.INITIALIZING.name
    assert payload["processed_frame_count"] == 0
    assert payload["collection_progress_percent"] == 0.0


def test_status_reflects_the_supervisor_snapshot(
    bound: WebController, supervisor: FakeSupervisor, client: TestClient
) -> None:
    supervisor.status = make_status(collected=25, target=100, frames=25)

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
    controller.trigger_alarm(make_alarm_event(score=-0.10, timestamp=1_700_000_000_000))
    controller.trigger_alarm(make_alarm_event(score=-0.90, timestamp=1_700_000_005_000))

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
        controller.trigger_alarm(make_alarm_event(timestamp=1_700_000_000_000 + i))

    alarms = client.get("/alerts").json()["alarms"]
    assert len(alarms) == 20
    # The five oldest were evicted.
    assert alarms[-1]["timestamp"] == 1_700_000_000_005


def test_alarm_frame_endpoint_returns_jpeg(
    controller: WebController, client: TestClient
) -> None:
    controller.trigger_alarm(make_alarm_event())
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


def test_mjpeg_stream_frames_and_stops_on_disconnect(
    bound: WebController, supervisor: FakeSupervisor
) -> None:
    """Exercised at generator level rather than over HTTP.

    The MJPEG response is endless by design, so an HTTP-level test can only
    read a chunk and abandon the connection - it can never assert the loop
    *ends*. Terminating on client disconnect is the whole point (otherwise
    every closed browser tab leaks a spinning task), so that's asserted here
    directly.
    """
    supervisor.latest_frame = FRAME
    request = FakeRequest(disconnect_after=2)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in bound._mjpeg_stream(request)]

    chunks = asyncio.run(collect())

    # Two frames yielded, then the disconnect ended the loop rather than
    # streaming forever.
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.startswith(b"--frame\r\n")
        assert b"Content-Type: image/jpeg" in chunk
        assert b"\xff\xd8" in chunk  # JPEG SOI marker in the payload


def test_mjpeg_stream_waits_without_yielding_before_first_frame(
    bound: WebController,
) -> None:
    """No frame produced yet must not emit an empty/corrupt JPEG part."""
    request = FakeRequest(disconnect_after=2)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in bound._mjpeg_stream(request)]

    assert asyncio.run(collect()) == []


def test_a_frame_is_encoded_once_however_often_it_is_served(
    bound: WebController, supervisor: FakeSupervisor
) -> None:
    """Several viewers polling faster than the loop produces frames must not
    each pay for their own encode."""
    supervisor.latest_frame = FRAME

    first = bound._jpeg_for(FRAME)
    again = bound._jpeg_for(FRAME)

    assert first is again  # the identical bytes object, not just an equal one

    # A different frame is a different encode.
    other = bound._jpeg_for(np.zeros((32, 32, 3), dtype=np.uint8))
    assert other is not first

# --- Control endpoints ----------------------------------------------------------------
#
# The controller never acts itself: it forwards requests to the Supervisor,
# which applies them on its own thread. So these assert forwarding, not
# behaviour - the behaviour lives in test_supervisor_controls.py.


@pytest.fixture
def control() -> FakeSupervisor:
    return FakeSupervisor()


@pytest.fixture
def wired(control: FakeSupervisor) -> TestClient:
    controller = WebController(
        cameras_provider=lambda active: [
            CameraOption(id=0, name="Integrated Webcam"),
            CameraOption(id=1, name="Logitech C920"),
        ]
    )
    controller.bind_supervisor(control)
    return TestClient(controller.app)


def test_control_endpoints_are_503_without_a_supervisor(client: TestClient) -> None:
    """A dashboard with nothing bound is read-only, not broken."""
    assert client.post("/api/control/start").status_code == 503
    assert client.post("/api/control/stop").status_code == 503
    assert client.post("/api/control/camera", json={"camera": 1}).status_code == 503


def test_start_and_stop_are_forwarded(wired: TestClient, control: FakeSupervisor) -> None:
    assert wired.post("/api/control/start").status_code == 200
    assert control.starts == 1

    assert wired.post("/api/control/stop").status_code == 200
    assert control.stops == 1


def test_camera_selection_is_forwarded_as_an_index(
    wired: TestClient, control: FakeSupervisor
) -> None:
    """A <select> sends strings; an OpenCV index has to arrive as an int."""
    assert wired.post("/api/control/camera", json={"camera": "1"}).status_code == 200
    assert control.cameras == [1]


def test_camera_selection_keeps_urls_as_strings(
    wired: TestClient, control: FakeSupervisor
) -> None:
    url = "rtsp://user:pass@host/stream"
    assert wired.post("/api/control/camera", json={"camera": url}).status_code == 200
    assert control.cameras == [url]


def test_camera_selection_rejects_nonsense(wired: TestClient, control: FakeSupervisor) -> None:
    assert wired.post("/api/control/camera", json={}).status_code == 422
    assert wired.post("/api/control/camera", json={"camera": ""}).status_code == 422
    assert control.cameras == []


def test_camera_list_is_served(wired: TestClient) -> None:
    payload = wired.get("/api/cameras").json()
    assert payload["cameras"] == [
        {"id": 0, "name": "Integrated Webcam"},
        {"id": 1, "name": "Logitech C920"},
    ]


def test_camera_list_survives_a_failing_provider() -> None:
    """Enumeration touches hardware and can fail; the page must still load."""

    def boom(active):
        raise RuntimeError("no such device")

    client = TestClient(WebController(cameras_provider=boom).app)
    assert client.get("/api/cameras").json()["cameras"] == []


def test_status_reports_camera_and_error(
    client: TestClient, bound: WebController, supervisor: FakeSupervisor
) -> None:
    supervisor.status = (
        SystemStatus(
            state=SupervisorState.IDLE,
            processed_frame_count=0,
            collection_progress=0.0,
            collected_count=0,
            collection_target=1000,
            total_alarms=0,
            camera=1,
            error="Could not open camera 1.",
        )
    )
    payload = client.get("/api/status").json()
    assert payload["state"] == "IDLE"
    assert payload["camera"] == 1
    assert payload["error"] == "Could not open camera 1."


def test_a_new_run_hides_the_previous_runs_alarms(
    client: TestClient, bound: WebController, supervisor: FakeSupervisor
) -> None:
    """Alarms scored against the old baseline mean nothing against the new
    one. The run id says so; nobody tells the panel to forget them."""
    supervisor.status = make_status(run_id=4)
    bound.trigger_alarm(make_alarm_event())
    assert len(client.get("/alerts").json()["alarms"]) == 1

    supervisor.status = make_status(run_id=5)

    assert client.get("/alerts").json()["alarms"] == []
