package com.lovesoft.winner.web;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.lovesoft.winner.engine.GameManager;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Starts the real HTTP server on a free port and exercises the REST API end to end. */
class GameApiIntegrationTest {

    private ChessHttpServer server;
    private HttpClient client;
    private String base;

    @BeforeEach
    void startServer() throws Exception {
        server = new ChessHttpServer("127.0.0.1", 0, new GameManager());
        server.start();
        client = HttpClient.newBuilder().proxy(HttpClient.Builder.NO_PROXY).build();
        base = "http://127.0.0.1:" + server.getPort();
    }

    @AfterEach
    void stopServer() {
        server.stop(0);
    }

    private HttpResponse<String> get(String path) throws Exception {
        return client.send(HttpRequest.newBuilder(URI.create(base + path)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> post(String path, String json) throws Exception {
        return client.send(HttpRequest.newBuilder(URI.create(base + path))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(json)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private static JsonObject json(HttpResponse<String> response) {
        return JsonParser.parseString(response.body()).getAsJsonObject();
    }

    @Test
    void fullGameFlow() throws Exception {
        HttpResponse<String> start = post("/api/game/start", "{\"white\":\"Alice\"}");
        assertEquals(201, start.statusCode());
        String id = json(start).get("gameId").getAsString();
        assertEquals("Alice", json(start).getAsJsonObject("state").get("whitePlayer").getAsString());

        HttpResponse<String> state = get("/api/game/" + id + "/state");
        assertEquals(200, state.statusCode());
        assertEquals("IN_PROGRESS", json(state).get("status").getAsString());
        assertEquals("WHITE", json(state).get("turn").getAsString());
        assertTrue(json(state).has("resultReason"), "nulls are serialised");
        assertEquals("no-store", state.headers().firstValue("Cache-Control").orElse(""));

        HttpResponse<String> move = post("/api/game/" + id + "/move", "{\"from\":\"e2\",\"to\":\"e4\"}");
        assertEquals(200, move.statusCode());
        assertEquals("BLACK", json(move).get("turn").getAsString());

        HttpResponse<String> illegal = post("/api/game/" + id + "/move", "{\"from\":\"e2\",\"to\":\"e4\"}");
        assertEquals(422, illegal.statusCode());
        assertTrue(json(illegal).has("error"));

        post("/api/game/" + id + "/move", "{\"san\":\"e5\"}");
        post("/api/game/" + id + "/move", "{\"san\":\"Bc4\"}");
        post("/api/game/" + id + "/move", "{\"san\":\"Nc6\"}");
        post("/api/game/" + id + "/move", "{\"san\":\"Qh5\"}");
        post("/api/game/" + id + "/move", "{\"san\":\"Nf6\"}");
        HttpResponse<String> mate = post("/api/game/" + id + "/move", "{\"san\":\"Qxf7#\"}");
        assertEquals("FINISHED_WHITE_WON", json(mate).get("status").getAsString());
        assertEquals("CHECKMATE", json(mate).get("resultReason").getAsString());

        HttpResponse<String> afterMate = post("/api/game/" + id + "/move", "{\"san\":\"a6\"}");
        assertEquals(409, afterMate.statusCode());
    }

    @Test
    void errorStatuses() throws Exception {
        assertEquals(404, get("/api/game/" + UUID.randomUUID() + "/state").statusCode());
        assertEquals(400, get("/api/game/not-a-uuid/state").statusCode());
        assertEquals(405, get("/api/game/start").statusCode());
        assertEquals(404, get("/api/game/whatever").statusCode());

        String id = json(post("/api/game/start", "")).get("gameId").getAsString();
        assertEquals(405, get("/api/game/" + id + "/move").statusCode());
        assertEquals(400, post("/api/game/" + id + "/move", "{not json").statusCode());
        assertEquals(400, post("/api/game/" + id + "/move", "").statusCode());
        assertEquals(422, post("/api/game/" + id + "/move", "{}").statusCode());
    }

    @Test
    void servesStaticFiles() throws Exception {
        HttpResponse<String> index = get("/");
        assertEquals(200, index.statusCode());
        assertTrue(index.headers().firstValue("Content-Type").orElse("").startsWith("text/html"));
        assertTrue(index.body().contains("chessboard"));

        assertEquals(200, get("/app.js").statusCode());
        assertEquals(200, get("/style.css").statusCode());
        assertEquals(404, get("/missing.txt").statusCode());
    }
}
