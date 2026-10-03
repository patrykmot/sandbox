I want to build a lightweight, web-based Chess application. Please act as an expert Full-Stack Java Developer and write the complete codebase for this project based on the requirements below.

### 1. High-Level Architecture
- **Backend:** Java (JDK 11+). Strictly use the built-in `com.sun.net.httpserver.HttpServer` for both exposing REST APIs and serving static web files. Do NOT use Spring Boot, Javalin, or Tomcat. 
- **Core Game Engine:** Use the `bhlangonijr/chesslib` library (Maven coordinates: `com.github.bhlangonijr:chesslib:1.3.7` via JitPack) as the foundation for move generation, FEN/PGN parsing, and move validation.
- **Frontend:** HTML, Bootstrap, and jQuery. Use `chessboard.js` (or a similar lightweight jQuery chessboard library) to beautifully render the chessboard, the pieces, and handle UI drag-and-drop. 
- **Startup:** When the Java `main` method runs, it should initialize the game engine manager, start the HTTP server on a specified port (e.g., 8080), and then automatically open the user's default web browser to `http://localhost:8080` using `java.awt.Desktop`.

### 2. Backend Requirements & State Machine
- **Game Engine Isolation:** The backend must be completely decoupled from the UI. The UI will communicate via HTTP endpoints. This is critical because in the future, the backend will be used to automatically play out PGN files (Portable Game Notation) headlessly.
- **Concurrency (Multiple Games):** The backend must support multiple games simultaneously. Implement a `GameManager` that stores games in a thread-safe `ConcurrentHashMap` with a unique Game ID (UUID).
- **Player Abstraction:** Create an abstract `Player` concept with a generic `makeMove()` or equivalent contract. Provide a `HumanPlayer` implementation for now. (A `BotPlayer` will be added later). Currently, a game will just consist of two `HumanPlayer`s taking turns.
- **State Machine:** Implement a simple state machine for the game: `WAITING_TO_START`, `IN_PROGRESS`, `FINISHED_WHITE_WON`, `FINISHED_BLACK_WON`, `FINISHED_DRAW`.
- **API Endpoints Required:**
  - `POST /api/game/start` -> Returns a new Game ID.
  - `GET /api/game/{id}/state` -> Returns the current board state (FEN string), whose turn it is, player names, and the state machine status.
  - `POST /api/game/{id}/move` -> Accepts a move (in SAN or from-to squares), validates it via `chesslib`, applies it to the state machine, and returns the updated state.

### 3. Frontend Requirements
- A clean Bootstrap layout showing the current Game ID, the Player Names (e.g., "Player 1 (White)" vs "Player 2 (Black)"), and the game status.
- Render the chessboard nicely. When a user drags and drops a piece, the frontend should immediately send an AJAX `POST` request to `/api/game/{id}/move`. 
- If the move is invalid (backend returns an error), the frontend piece should snap back to its original square.
- Use simple polling (`setInterval` in jQuery) every 1-2 seconds to hit `/api/game/{id}/state` so that if one player moves in a different browser tab, the other tab updates automatically.

### 4. What You Need to Provide
Please generate the full, working codebase organized into files, including:
1. `pom.xml` (with JitPack repository and `chesslib` dependency).
2. The complete Java backend source code (Main class, HttpServer setup, Static File Handler, REST Handlers, GameManager, Game/Player models).
3. The frontend files (`index.html`, `app.js`, `style.css`) to be placed in the backend's resources directory.
4. Brief instructions on how to compile and run the application.

Ensure the code is robust, well-commented, and ready to be pasted into an IDE.