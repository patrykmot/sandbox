"""Implementation 1 of IVideoEncoder: Ultralytics YOLO + ByteTrack."""

from __future__ import annotations

import time

import numpy as np
from ultralytics import YOLO

from src.interfaces.encoder import IVideoEncoder, StateVector


class YOLOVideoEncoder(IVideoEncoder):
    """Detects and tracks objects with Ultralytics YOLO, mapping the result to
    StateVectors.

    Velocity (vx, vy) is derived from the movement of the same tracked object
    (by ByteTrack track id) between consecutive calls to `encode`. A track's
    first appearance has no prior position, so its velocity is reported as
    (0.0, 0.0).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._tracker = tracker
        # track_id -> (x, y, t_ms) of that object's last observed position.
        self._last_seen: dict[int, tuple[float, float, int]] = {}

    def encode(self, frame: np.ndarray) -> list[StateVector]:
        now_ms = time.time_ns() // 1_000_000

        results = self._model.track(
            frame,
            conf=self._confidence,
            tracker=self._tracker,
            persist=True,
            verbose=False,
        )

        state_vectors: list[StateVector] = []
        if not results:
            return state_vectors

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            # No tracked detections in this frame (nothing above threshold,
            # or ByteTrack hasn't assigned ids yet).
            return state_vectors

        xywh = boxes.xywh.cpu().numpy()  # center-x, center-y, width, height
        track_ids = boxes.id.cpu().numpy().astype(int)
        classes = boxes.cls.cpu().numpy().astype(int)

        for (cx, cy, w, h), track_id, cls_id in zip(xywh, track_ids, classes):
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            size = w * h

            prev = self._last_seen.get(int(track_id))
            if prev is None:
                vx, vy = 0.0, 0.0
            else:
                prev_x, prev_y, prev_t = prev
                dt_seconds = (now_ms - prev_t) / 1000.0
                if dt_seconds > 0:
                    vx = (cx - prev_x) / dt_seconds
                    vy = (cy - prev_y) / dt_seconds
                else:
                    vx, vy = 0.0, 0.0

            self._last_seen[int(track_id)] = (cx, cy, now_ms)

            state_vectors.append(
                StateVector(
                    t=now_ms,
                    x=cx,
                    y=cy,
                    vx=vx,
                    vy=vy,
                    size=size,
                    object_type=int(cls_id),
                )
            )

        return state_vectors
