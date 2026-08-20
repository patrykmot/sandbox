PROMPT: Modular Edge Video Anomaly Detection System

---
---
---

**General Note:** If you see something is unclear or brake logical rules so you cant figure it out, just ask me for question. 

# Phase I
## Objective
Act as a Principal Python Software Engineer. Write a modular, production-ready, clean, and well-typed Python 3.14 application for real-time video anomaly detection on edge devices.
The system converts video streams into mathematical vectors called StateVector (timestamp in milliseconds, position 2D, speed 2d, size 1D, object type 1D) 
Next, StateVector list is passed to featureEncoder to get FeatureVector list. FeatureVector is vector with floats N sized and timestamp for which this feature happened. Implementation of featureEncoder will be done in Phase II.
Whole idea is to catch multiple different objects (and they relations) at same time, and find anomaly for it later when training is finished.


---

## Technical Guidelines & Constraints
* Language: Python 3.12 (Use modern type hinting, typing.Protocol or abc.ABC, dataclasses, and native asyncio/threading where applicable).
* Architecture Design: Interface-driven design (Strategy Pattern). Components 1, 2, 3, 5 AND 6 MUST be defined via strict Abstract Base Classes (ABCs) or Protocols to allow seamless swapping of implementations.
* Dependencies: opencv-python, ultralytics, scikit-learn, fastapi, uvicorn, numpy, pydantic, pydantic-settings.
 Testing: Write JUNIT tests where feasible (test will be easy to maintain, and fast to launch). Use mocking so no camera is needed to run it.

---

## System Architecture & Data Flow

[ 1. Video Source ] ──> (Frame) ──> [ 2. Video Encoder ]
                                             │
                                   (list[StateVector])
                                             ▼
                                   [ 6. Feature Encoder ]
                                             │
                                  (list[FeatureVector])
                                             ▼
 [ 5. Alarm Handler ] <── (Alarm) ── [ 4. Supervisor ] <──> [ 3. AI Detector ]
          │                                  │
          └─────────── [ Phase II Web UI ] ◄─┘
          

---

## A. Component Specifications & Interfaces (1,2,3,5,6) - Those are domains of application.

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
  - StateVector: Data class containing:
    - t: timestamp in milliseconds ( Unix epoch timestamp in milliseconds, UTC), just to know when this state happened, and it can be passed to FeatureVector
    - x: float (2D Center X)
    - y: float (2D Center Y)
    - Vx: float speed on X (pixels change / second)
    - Vy: float speed on Y (pixels change / second)
    - size: float: bounding box surface area
    - object_type: int (1D Class ID, e.g., 0 for Person)
* Methods:
  - encode(frame: np.ndarray) -> list[StateVector]: Processes frame and outputs detected object vectors.
* Implementation 1 (YOLOVideoEncoder):
  - Uses Ultralytics YOLO (yolov8n.pt) to detect objects.
  - Maps outcome of detection to StateVector. Base on tracking id calculate speed vector (Vx,Vy) as well. Please use ByteTrack for tracking. 

### Interface 3: Anomaly Detector (IAnomalyDetector)
Abstract interface for training and predicting anomalies from vectors.
* Methods:
  - fit(training_data: list[list[FeatureVector]]) -> None: Trains the underlying model on collected vectors.
  - predict(vector: list[FeatureVector]) -> tuple[bool, float]: Returns (is_anomaly, anomaly_score).
  - is_trained() -> bool: Returns model status.
  - save(path: str) -> None / load(path: str) -> None: Persistence methods.
* Implementation 1 (IsolationForestAnomalyDetector):
  - Uses sklearn.ensemble.IsolationForest.
  - Configurable contamination parameter (e.g., 0.05).
  - One Isolation Forest for the entire data.

### Interface 5: Alarm Handler (IAlarmHandler)
Abstract interface to dispatch notifications when an anomaly is detected.
* Data Structure:
  - AlarmEvent: Timestamp, vector, frame snapshot (optional), anomaly score, and description.
* Methods:
  - trigger_alarm(event: AlarmEvent) -> None: Handles the alarm payload.
* Implementation 1 (ConsoleLoggerAlarmHandler):
  - Logs the alarm with high visual visibility to stdout/stderr.
  - Stores recent alarm history in memory for Web UI access.

### Interface 6: Feature Encoder (IFeatureEncoder)
Abstract interface to that maps from StateVector into FeatureVector
* Data Structure: 
  - FeatureVector: Data class containing 
    - t - same as StateVector.t
    - vector - contain N float features (which describe vector values)
* Methods:
  - encode(list[StateVector]) -> list[FeatureVector]: Process StateVector and produce  FeatureVector that could be used to train or feed ML model to detect anomalies. Outcome list size do not need to be same as income list.
* Use Feature Encoder output to feed Anomaly Detector. One vector FeatureVector.vector its one fixed size vector that you should feed isolation forest fit. Of course for predict you need to have list of FeatureVectors.
* In Phase I implementation is empty. Will be added in Phase II. For now you can just add DummyFeatureEncoder.

---

## B. Separated description for Component 4: System Supervisor (Supervisor) - Application itself

The Coordinator that manages lifecycle, state machine transitions, and component execution loops.

### State Machine States:
1. INITIALIZING: Setting up sources and models.
2. COLLECTING_DATA: Accumulating normal baseline vectors for N days (or N frames/samples for testing). The collection phase assumes that the environment is known to be normal. No anomaly detection is performed during collection.
3. TRAINING: Triggers model training when data requirement is reached. This is global model training with all collected training data.
4. MONITORING: Active execution phase. Feeds live encoded vectors to the model.
5. ALARM_ACTIVE: Triggered when predict() flags an anomaly. Invokes IAlarmHandler.trigger_alarm().
6. ERROR: Fault state.

### Core Loop Logic:
1. Read frame from IVideoSource.
2. Extract vectors via IVideoEncoder.
3. Extract features via IFeatureEncoder.
4. Depending on state:
   - If COLLECTING_DATA: Append vectors to training dataset. Track collection progress (e.g., % complete).
   - If collection target reached: Transition to TRAINING -> call IAnomalyDetector.fit() -> transition to MONITORING.
   - If MONITORING: Pass vectors to IAnomalyDetector.predict(). If is_anomaly == True, state transitions momentarily or concurrently triggers IAlarmHandler.


### Thread-Safety & Data Sharing Constraint:
* Use a thread-safe mechanism (e.g., `threading.Lock` combined with a shared instance variable, or an `asyncio.Queue`) to safely pass the latest annotated frame and system state from the Supervisor loop to the Web Server for the MJPEG stream.


## D. Suggested Directory Architecture

Please construct the project using the following layout:

src/
├── config.py                 # Configuration dataclasses / Pydantic settings
├── interfaces/
│   ├── __init__.py
│   ├── video_source.py       # IVideoSource
│   ├── encoder.py            # IVideoEncoder & StateVector + IFeatureEncoder & FeatureVector
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
└── main.py                   # Entry point (CLI & Startup)

---

## 5. Configuration Requirements (config.py)

Provide a single configuration file supporting environment variables or CLI overrides using `pydantic-settings`:
* COLLECTION_TARGET_VALUE: Numerical target 1000 vectors.
* CAMERA_INDEX: 0 or RTSP string.
* SERVER_HOST: "0.0.0.0"
* SERVER_PORT: 8000

---

## 6. Execution Requirements
1. Provide a main.py script that instantiates default implementations, passes them to Supervisor, and runs the application.
2. Provide a clean requirements.txt.
3. Include inline comments explaining interface decoupling so new hardware/software components can be plugged in by swapping implementation classes in main.py.

---
---
---