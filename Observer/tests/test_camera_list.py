"""Camera enumeration: probing, labelling and the short-lived cache.

No camera is needed - cv2.VideoCapture is replaced with a fake that answers
for a chosen set of indices, which is also the only way to test the "index
does not exist" path deterministically.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.implementations import camera_list
from src.implementations.camera_list import label_for, list_cameras
from src.interfaces.video_source import CameraOption


class FakeCapture:
    """Stands in for cv2.VideoCapture; `opened` decides which indices exist."""

    opened: set[int] = set()
    opens: list[int] = []

    def __init__(self, index: int) -> None:
        self._index = index
        FakeCapture.opens.append(index)
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self._index in FakeCapture.opened

    def release(self) -> None:
        self.released = True


@pytest.fixture(autouse=True)
def fresh_cache():
    """Each test starts with an empty probe cache and a clean fake."""
    camera_list._cache = None  # test-only introspection
    FakeCapture.opened = {0, 2}
    FakeCapture.opens = []
    yield
    camera_list._cache = None


def test_finds_the_indices_that_answer() -> None:
    with patch.object(camera_list.cv2, "VideoCapture", FakeCapture):
        assert list_cameras(max_probe=4) == [
            CameraOption(id=0, name="Camera 0"),
            CameraOption(id=2, name="Camera 2"),
        ]


def test_probes_every_index_up_to_the_limit() -> None:
    """Indices can have gaps, so the probe must not stop at the first miss."""
    with patch.object(camera_list.cv2, "VideoCapture", FakeCapture):
        list_cameras(max_probe=4)
    assert FakeCapture.opens == [0, 1, 2, 3]


def test_results_are_cached_so_a_page_load_costs_one_probe() -> None:
    with patch.object(camera_list.cv2, "VideoCapture", FakeCapture):
        list_cameras(max_probe=4)
        first_round = list(FakeCapture.opens)
        list_cameras(max_probe=4)
        list_cameras(max_probe=4)

    assert FakeCapture.opens == first_round  # no second probe


def test_force_refresh_probes_again() -> None:
    with patch.object(camera_list.cv2, "VideoCapture", FakeCapture):
        list_cameras(max_probe=2)
        list_cameras(max_probe=2, force_refresh=True)

    assert FakeCapture.opens == [0, 1, 0, 1]


def test_expired_cache_probes_again() -> None:
    with patch.object(camera_list.cv2, "VideoCapture", FakeCapture):
        list_cameras(max_probe=2)
        expires_at, max_probe, options = camera_list._cache
        camera_list._cache = (expires_at - camera_list.CACHE_TTL_SECONDS - 1, max_probe, options)
        list_cameras(max_probe=2)

    assert FakeCapture.opens == [0, 1, 0, 1]


def test_active_camera_is_kept_even_when_it_cannot_be_reopened() -> None:
    """A camera in use often refuses a second open - it must not disappear
    from the dropdown while it is the one being previewed."""
    FakeCapture.opened = {1}
    with patch.object(camera_list.cv2, "VideoCapture", FakeCapture):
        options = list_cameras(max_probe=3, always_include=0)

    assert CameraOption(id=0, name="Camera 0") in options


def test_active_camera_is_not_listed_twice() -> None:
    with patch.object(camera_list.cv2, "VideoCapture", FakeCapture):
        options = list_cameras(max_probe=3, always_include=0)

    assert [o.id for o in options] == [0, 2]


def test_a_stream_url_survives_as_its_own_label() -> None:
    url = "rtsp://user:pass@host/stream"
    with patch.object(camera_list.cv2, "VideoCapture", FakeCapture):
        options = list_cameras(max_probe=0, always_include=url)

    assert options == [CameraOption(id=url, name=url)]


def test_a_backend_that_raises_does_not_break_enumeration() -> None:
    """Probing touches hardware; the page must still get a list."""

    def boom(index: int):
        raise RuntimeError("backend exploded")

    with patch.object(camera_list.cv2, "VideoCapture", boom):
        assert list_cameras(max_probe=3) == []


def test_labels() -> None:
    assert label_for(0) == "Camera 0"
    assert label_for("rtsp://cam/stream") == "rtsp://cam/stream"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
