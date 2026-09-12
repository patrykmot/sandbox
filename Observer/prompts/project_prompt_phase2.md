# PHASE II
## Web Dashboard & HTTP Server (WebDashboard)
A lightweight FastAPI web server running concurrently with the processing loop.
Ask me any question if something is unlogical on not clear for you, or it will brake good engineering principles. 

## Strict Concurrency Requirement:
* The OpenCV/Supervisor processing loop MUST run in a separate background `threading.Thread` to prevent blocking the asynchronous FastAPI event loop.

## Features:
  - Simple HTML/JS dashboard embedded or served via single file (use Bootstrap/ JQuery / CSS).
  - Live View: Displays live processed camera feed (MJPEG stream endpoint).
  - System State Indicator: Shows current state (COLLECTING_DATA, MONITORING, ALARM, System stats etc.).
  - Progress Bar: Shows data collection progress.
  - Alarm Panel:
    - Table displaying recent alarm events with timestamps and Camera Frame View CFV (only last x=20 values)
    - CFV - should contain borders (use YOLO result.plot() function) - only for UI

## Architecture
  - Create new interface IController 
    - Supervisor with get this IController implementation, and it will pass it over all needed data to be displayed in UI Web page.
    - First implementation is IWebController which will be simple FastAPI server.
    - In future IController can signal some actions for Supervisor (like stop training, or restart process, pause itp...). Do not implement this part right now.

## Endpoints for web server:
  - GET /: UI Web page.
  - GET /api/status: Returns JSON with state, processed frame count, collection progress %, total alarms.
  - GET /video_feed: MJPEG stream endpoint for real-time visualization.
  - GET /alerts - Return data needed to DISPLAY Alarm Panel
  - Other endpoints if needed.
