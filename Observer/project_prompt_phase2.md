# PHASE II  <--- AS AI SKIP THIS PART NOW!!!!!!!!!!!
## Web Dashboard & HTTP Server (WebDashboard)
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
  - 5. Expose current frame, system stats, current state, and logs to the Web Dashboard.