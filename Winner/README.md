# Winner Chess!

Lightweight web chess: a plain-JDK `HttpServer` backend with the
[chesslib](https://github.com/bhlangonijr/chesslib) engine, and a Bootstrap + jQuery + chessboard.js frontend.

## Requirements

- JDK 17+ (`java` on the PATH or `JAVA_HOME` set)
- No Maven install needed: the Maven Wrapper (`mvnw` / `mvnw.cmd`) downloads Maven 3.9.11 on first use
  into `%USERPROFILE%\.m2\wrapper` (Windows) or `~/.m2/wrapper` (macOS/Linux)
- Internet access on first build (Maven Central + JitPack) and in the browser (CDN libraries, piece images)

## Build and run

Windows (Command Prompt or PowerShell):

```bat
mvnw.cmd clean package
java -jar target\chess-web-1.0.0.jar
```

macOS / Linux / Git Bash:

```bash
./mvnw clean package                      # compiles, runs tests, builds the fat JAR
java -jar target/chess-web-1.0.0.jar      # starts on http://localhost:8080 and opens the browser
```

Any Maven command works through the wrapper, e.g. `mvnw.cmd test` or `mvnw.cmd compile exec:java`.

Options:

| Option | Example | Default |
| --- | --- | --- |
| Port (argument) | `java -jar target/chess-web-1.0.0.jar 9090` | `8080` |
| Port (property) | `java -Dchess.port=9090 -jar ...` | `8080` |
| Bind address | `java -Dchess.host=127.0.0.1 -jar ...` | `0.0.0.0` (reachable on your LAN) |
| Don't open a browser | `java -jar ... --no-browser` | browser opens |

From an IDE: run `com.lovesoft.winner.app.ChessApplication`. With the wrapper: `mvnw.cmd compile exec:java`.

## Playing

The page creates a game and puts its id in the URL (`/?game=<uuid>`). Open the same link in a second tab or
browser to play the other side; add `&side=black` to view the board from Black's side. Every drag is validated
by the backend; illegal moves snap back. Tabs poll the server every second, so moves show up in the other tab
automatically. Pawns promote to a queen when dragged to the last rank.

## REST API

| Method + path | Body | Success | Errors |
| --- | --- | --- | --- |
| `POST /api/game/start` | optional `{"white":"Alice","black":"Bob"}` | `201 {gameId, state}` | `405` |
| `GET /api/game/{id}/state` | - | `200 state` | `400` bad id, `404` unknown game |
| `POST /api/game/{id}/move` | `{"from":"e2","to":"e4","promotion":"q"}` or `{"san":"Nf3"}` | `200 state` | `400` bad JSON, `404`, `409` game finished, `422` illegal move |

State object:

```json
{
  "gameId": "6f1c...",
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "turn": "BLACK",
  "status": "IN_PROGRESS",
  "whitePlayer": "Player 1",
  "blackPlayer": "Player 2",
  "lastMove": { "from": "e2", "to": "e4", "san": "e4" },
  "moveHistory": ["e4"],
  "inCheck": false,
  "resultReason": null
}
```

`status` is one of `WAITING_TO_START`, `IN_PROGRESS`, `FINISHED_WHITE_WON`, `FINISHED_BLACK_WON`, `FINISHED_DRAW`.
`resultReason` (finished games only): `CHECKMATE`, `STALEMATE`, `INSUFFICIENT_MATERIAL`, `THREEFOLD_REPETITION`,
`FIFTY_MOVE_RULE`.

Quick check with curl:

```bash
curl -s -X POST localhost:8080/api/game/start
curl -s -X POST localhost:8080/api/game/<id>/move -H "Content-Type: application/json" -d '{"san":"e4"}'
curl -s localhost:8080/api/game/<id>/state
```

## Code layout

```
src/main/java/com/lovesoft/winner/
  app/ChessApplication.java     main(): GameManager -> HTTP server -> browser
  engine/                       pure Java + chesslib, no HTTP/JSON
    GameManager.java            ConcurrentHashMap<UUID, Game>
    Game.java                   Board, players, state machine, per-game lock
    GameStatus.java, ResultReason.java, MoveRequest.java, GameSnapshot.java, exceptions
    player/Player.java          abstract seat: makeMove(snapshot) -> Optional<MoveRequest>
    player/HumanPlayer.java     returns empty: waits for an external move
  web/                          HTTP + JSON only
    ChessHttpServer.java        HttpServer, contexts, worker pool
    GameApiHandler.java         /api/game routes and error mapping
    StaticFileHandler.java      serves src/main/resources/web
    Json.java, HttpSupport.java, BadRequestException.java
src/main/resources/web/         index.html, app.js, style.css
src/test/java/...               engine unit tests + HTTP integration test
```

## Extending

- **Bots:** subclass `Player`, return a move from `makeMove(snapshot)` and `false` from `acceptsExternalMoves()`.
  `Game` lets automatic players move right after every move (and on `start()`), so no API change is needed.
  Register it with `gameManager.createGame(white, black)` and call `game.start()`.
- **Headless PGN replay:** create a game through `GameManager`, then feed each SAN move with
  `game.submitMove(MoveRequest.ofSan(san))`; the engine never touches HTTP.
