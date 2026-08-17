# PROMPT: Modular Edge Video Anomaly Detection System

## Objective
Act as a Principal Python Software Engineer. Write a modular, production-ready, clean, and well-typed Python 3.14 application for real-time video anomaly detection on edge devices.
The system converts video streams into low-dimensional mathematical vectors (position 2D, size 1D, object type 1D) and feeds them into an unsupervised machine learning model (Isolation Forest) to detect temporal/spatial anomalies.

---

## Technical Guidelines & Constraints
* Language: Python 3.14 (Use modern type hinting, typing.Protocol or abc.ABC, dataclasses, and native asyncio/threading where applicable).
* Architecture Design: Interface-driven design (Strategy Pattern). Components 1, 2, 3, and 5 MUST be defined via strict Abstract Base Classes (ABCs) or Protocols to allow seamless swapping of implementations.
* Dependencies: opencv-python, ultralytics, scikit-learn, fastapi, uvicorn, numpy, pydantic, pydantic-settings.

---

## System Architecture & Data Flow

 [ 1. Video Source ] ──> (Frame) ──> [ 2. Video Encoder ]
                                             │
                                       (Vector [X,Y,S,T])
                                             │
                                             ▼
 [ 5. Alarm Handler ] <── (Alarm) ── [ 4. Supervisor ] <──> [ 3. AI Detector ]
          │                                  │
          └──────────── [ 6. Web Dashboard ] ◄┘

---

## A. Component Specifications & Interfaces (1,2,3,5)

### Interface 1: Video Data Source (IVideoSource)
Abstract interface to stream frames from hardware or software sources.
* Methods:
  - get_frame() -> tuple[bool, np.ndarray | None]: Returns success flag and frame array.
  - release() -> None: Cleans up resources.
* Implementation 1 (CameraVideoSource):
  - Captures live stream from a camera using OpenCV (cv2.VideoCapture).
  - Supports camera index or RTSP stream URL via configuration.

### Interface 2: Video Encoder (IVideoEncoder)
Abstract interface to extract features from frames and convert them into low-dimensional vectors.
* Data Structure:
  - ObjectVector: Data class containing:
    - x: float (2D Center X)
    - y: float (2D Center Y)
    - size: float (1D Area or Bounding Box Scale)
    - object_type: int (1D Class ID, e.g., 0 for Person)
    - to_array() -> list[float]: Returns numeric 1D list [x, y, size, object_type].
* Methods:
  - encode(frame: np.ndarray) -> list[ObjectVector]: Processes frame and outputs detected object vectors.
* Implementation 1 (YOLOVideoEncoder):
  - Uses Ultralytics YOLO (yolov8n.pt) to detect objects.
  - Maps bounding boxes [x_center, y_center, width * height, class_id] to ObjectVector.

### Interface 3: Anomaly Detector (IAnomalyDetector)
Abstract interface for training and predicting anomalies from vectors.
* Methods:
  - fit(training_data: list[list[float]]) -> None: Trains the underlying model on collected vectors.
  - predict(vector: list[float]) -> tuple[bool, float]: Returns (is_anomaly, anomaly_score).
  - is_trained() -> bool: Returns model status.
  - save(path: str) -> None / load(path: str) -> None: Persistence methods.
* Implementation 1 (IsolationForestAnomalyDetector):
  - Uses sklearn.ensemble.IsolationForest.
  - Configurable contamination parameter (e.g., 0.05).

### Interface 5: Alarm Handler (IAlarmHandler)
Abstract interface to dispatch notifications when an anomaly is detected.
* Data Structure:
  - AlarmEvent: Timestamp, vector, frame snapshot (optional), anomaly score, and description.
* Methods:
  - trigger_alarm(event: AlarmEvent) -> None: Handles the alarm payload.
* Implementation 1 (ConsoleLoggerAlarmHandler):
  - Logs the alarm with high visual visibility to stdout/stderr.
  - Stores recent alarm history in memory for Web UI access.

---

## B. Separated description for Component 4: System Supervisor (Supervisor)

The Coordinator that manages lifecycle, state machine transitions, and component execution loops.

### State Machine States:
1. INITIALIZING: Setting up sources and models.
2. COLLECTING_DATA: Accumulating normal baseline vectors for N days (or N frames/samples for testing).
3. TRAINING: Triggers model training when data requirement is reached.
4. MONITORING: Active execution phase. Feeds live encoded vectors to the model.
5. ALARM_ACTIVE: Triggered when predict() flags an anomaly. Invokes IAlarmHandler.trigger_alarm().
6. ERROR: Fault state.

### Core Loop Logic:
1. Read frame from IVideoSource.
2. Extract vectors via IVideoEncoder.
3. Depending on state:
   - If COLLECTING_DATA: Append vectors to training dataset. Track collection progress (e.g., % complete).
   - If collection target reached: Transition to TRAINING -> call IAnomalyDetector.fit() -> transition to MONITORING.
   - If MONITORING: Pass vectors to IAnomalyDetector.predict(). If is_anomaly == True, state transitions momentarily or concurrently triggers IAlarmHandler.
4. Expose current frame, system stats, current state, and logs to the Web Dashboard.

### Thread-Safety & Data Sharing Constraint:
* Use a thread-safe mechanism (e.g., `threading.Lock` combined with a shared instance variable, or an `asyncio.Queue`) to safely pass the latest annotated frame and system state from the Supervisor loop to the Web Server for the MJPEG stream.

---

## C. Web Dashboard & HTTP Server (WebDashboard)

A lightweight FastAPI web server running concurrently with the processing loop.

### Strict Concurrency Requirement:
* The OpenCV/Supervisor processing loop MUST run in a separate background `threading.Thread` to prevent blocking the asynchronous FastAPI event loop.

* Features:
  - Simple HTML/JS dashboard embedded or served via single file.
  - Live View: Displays live processed camera feed (MJPEG stream endpoint).
  - System State Indicator: Shows current state (COLLECTING_DATA, MONITORING, ALARM, etc.).
  - Progress Bar: Shows data collection progress during Phase 1.
  - Alarm Log: Table displaying recent alarm events with timestamps and vector values.
* Endpoints:
  - GET /: UI Web page.
  - GET /api/status: Returns JSON with state, processed frame count, collection progress %, total alarms.
  - GET /video_feed: MJPEG stream endpoint for real-time visualization.

---

## D. Suggested Directory Architecture

Please construct the project using the following layout:

src/
├── config.py                 # Configuration dataclasses / Pydantic settings
├── interfaces/
│   ├── __init__.py
│   ├── video_source.py       # IVideoSource
│   ├── encoder.py            # IVideoEncoder & ObjectVector
│   ├── detector.py           # IAnomalyDetector
│   └── alarm.py              # IAlarmHandler & AlarmEvent
├── implementations/
│   ├── __init__.py
│   ├── camera_source.py      # CameraStreamSource
│   ├── yolo_encoder.py       # YOLOVectorEncoder
│   ├── iso_forest_detector.py# IsolationForestDetector
│   └── console_alarm.py      # ConsoleLoggerAlarmHandler
├── core/
│   ├── __init__.py
│   └── supervisor.py         # Supervisor (State Machine Logic)
├── web/
│   ├── __init__.py
│   └── server.py             # FastAPI App & MJPEG Streamer
├── templates/
│   └── index.html            # Web Dashboard Frontend
└── main.py                   # Entry point (CLI & Startup)

---

## 5. Configuration Requirements (config.py)

Provide a single configuration file supporting environment variables or CLI overrides using `pydantic-settings`:
* COLLECTION_DURATION_MODE: "DAYS" or "FRAMES" (default "FRAMES" with 1000 samples for quick test/demo).
* COLLECTION_TARGET_VALUE: Numerical target (e.g., 3 days or 5000 vectors).
* CAMERA_INDEX: 0 or RTSP string.
* MODEL_CONTAM_RATE: 0.05
* SERVER_HOST: "0.0.0.0"
* SERVER_PORT: 8000

---

## 6. Execution Requirements
1. Provide a main.py script that instantiates default implementations, passes them to Supervisor, starts the background Web Server, and runs the application.
2. Provide a clean requirements.txt.
3. Include inline comments explaining interface decoupling so new hardware/software components can be plugged in by swapping implementation classes in main.py.