# Observer — Modular Edge Video Anomaly Detection

Observer watches a live video stream, turns what it sees into numbers, learns what
"normal" looks like on its own, and raises an alarm when the scene stops matching
that baseline. It runs on a single machine (edge device), needs no labelled data,
and serves a live web dashboard while it works.

![Observer dashboard during the data-collection phase](Documents/screen.png)

---

## What it does

1. **Reads frames** from a camera or RTSP stream.
2. **Detects and tracks objects** with YOLOv8 + ByteTrack. Each tracked object
   becomes a `StateVector`: when it was seen, where it is, how fast it moves, how
   big it is, and what class it belongs to.
3. **Describes the whole frame as one vector.** All objects seen at the same moment
   are merged into a single `FeatureVector` — how many objects there are, how far
   apart they are, how fast they move, and how fast they move *relative to each
   other*. This is what makes multi-object **relations** learnable: "three people
   drifting apart slowly" and "three people converging fast" look very different,
   even though no single object is unusual.
4. **Learns the baseline.** The first N frames (default 1000) are collected, not
   judged. When the target is reached, an IsolationForest is trained on them.
   There is no dataset to prepare — whatever the camera normally sees becomes the
   definition of normal.
5. **Monitors.** Every later frame is scored, and anything the model calls an
   outlier raises an alarm carrying the score and a snapshot of the frame.
6. **Reports.** Alarms go to the console and to the dashboard: live view, current
   state, collection progress, and the most recent alarms with clickable frame
   thumbnails.

The system moves through these states:

`INITIALIZING → COLLECTING_DATA → TRAINING → MONITORING ⇄ ALARM_ACTIVE`
(plus `ERROR` if something in the loop fails).

---

## Data flow

```
[1. IVideoSource] ──frame──> [2. IVideoEncoder] ──list[StateVector]──> [6. IFeatureEncoder]
                                     │                                          │
                              annotated frame                          list[FeatureVector]
                                     │                                          ▼
                                     │                              [4. Supervisor] <──> [3. IAnomalyDetector]
                                     │                                     │
                                     └────> [7. IController — Web UI] <────┴──> [5. IAlarmHandler]
```

The annotated frame (boxes, labels, track ids) is for display only — the detection
model never sees it. Only the feature vectors reach the ML pipeline.

## Architecture

Interface-driven (Strategy pattern) throughout. Every component is an abstract base
class in `src/interfaces/`; the concrete classes live in `src/implementations/` and
are wired together in exactly one place — `src/main.py`. Nothing else knows which
implementation is in use, so swapping the camera for a video file, or IsolationForest
for another model, is a one-line change.

| # | Interface | Default implementation |
|---|-----------|------------------------|
| 1 | `IVideoSource` | `CameraVideoSource` — OpenCV camera or RTSP stream |
| 2 | `IVideoEncoder` | `YOLOVideoEncoder` — Ultralytics YOLO + ByteTrack |
| 3 | `IAnomalyDetector` | `IsolationForestAnomalyDetector` — scikit-learn |
| 4 | — (coordinator) | `Supervisor` — the state machine and per-frame loop |
| 5 | `IAlarmHandler` | `ConsoleLoggerAlarmHandler` |
| 6 | `IFeatureEncoder` | `FrameMergingFeatureEncoder` — one vector per frame |
| 7 | `IController` | `WebController` — FastAPI dashboard |

Dependencies point one way only: `implementations → core → interfaces`.

The Supervisor runs on a background thread while the web server owns the main
thread, and it only ever *pushes* data to the controller — it knows nothing about
how, or whether, any of it is displayed. A failing dashboard can't take the
processing loop down with it.

### Layout

```
src/
  main.py              where all the components are wired together
  config.py            every setting, overridable by env var or .env
  core/                the Supervisor: state machine + per-frame loop
  interfaces/          the seven abstract interfaces and the data they pass
  implementations/     the concrete components, plus dashboard.html
tests/                 pytest suite — no camera needed
prompts/               the original specification prompts
playground/            scratch scripts, not part of the running system
```

---

## Running it

```bash
pip install -r requirements.txt
python -m src.main            # dashboard on http://localhost:8000/
```

Needs Python 3.12+ and `src/yolov8n.pt` (included). During collection the console
shows a single updating progress bar; state changes are logged normally.

Everything is configured through environment variables or a `.env` file, for
example:

```bash
CAMERA_INDEX=rtsp://user:pass@host/stream   # a stream or video file instead of the webcam
COLLECTION_TARGET_VALUE=200                 # a shorter baseline while developing
SERVER_PORT=8080                            # dashboard port
```

See `src/config.py` for the full list — camera, dashboard, YOLO, detector and CSV
settings, all with working defaults, so it starts with no configuration at all.

### Dashboard

| Route | What it gives you |
|---|---|
| `/` | the dashboard page |
| `/api/status` | current state, frames processed, collection progress, alarm count |
| `/video_feed` | live MJPEG stream of the annotated camera view |
| `/alerts` | the most recent alarms, newest first |

### Feature CSV

Every feature vector is also appended to `feature_vectors.csv` (on by default) so
the data can be inspected offline, or tailed while the system runs. It is purely an
observation aid and never feeds anything back into the pipeline.

## Tests

```bash
pytest        # config lives in pytest.ini
pytest -s     # the walkthrough tests print their vectors with labels
```

No camera and no network required. The Supervisor tests drive the whole state
machine through mocks, the controller tests exercise the web routes and the live
stream, and the feature-encoder tests are written to be *read*, using
hand-checkable numbers rather than opaque fixtures.

## Extending it

Add a class implementing the relevant interface in `src/implementations/`, then
change the one constructor call in `src/main.py`. Typical cases:

- a video file or synthetic source instead of a camera (`IVideoSource`)
- e-mail, webhook or SMS alarms instead of the console (`IAlarmHandler`)
- a different model — autoencoder, one-class SVM (`IAnomalyDetector`)
- a different feature set (`IFeatureEncoder`)
- a headless run: pass no controller at all

## Known gaps

- **Trained models aren't saved yet.** Saving and loading are implemented and a
  path is configured, but nothing calls them, so every restart re-collects and
  re-trains from scratch.
- **The dashboard is read-only.** Pause, restart and stop-training controls were
  deliberately deferred; the interface was kept push-only so they can be added
  later without a redesign.
- **Two open questions in the detector**, marked as `TODO` in the code: how the
  training data is flattened, and whether loading a model should also restore the
  system state instead of starting a fresh collection phase.
