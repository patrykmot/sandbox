package com.lovesoft.winner.web;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/** Small helpers shared by the HTTP handlers. */
final class HttpSupport {

    /** Request bodies are tiny (a move or two names); anything bigger is rejected. */
    static final int MAX_BODY_BYTES = 16 * 1024;

    private HttpSupport() {
    }

    /** Reads the request body as UTF-8, enforcing {@link #MAX_BODY_BYTES}. */
    static String readBody(HttpExchange exchange) throws IOException {
        try (InputStream in = exchange.getRequestBody()) {
            byte[] bytes = in.readNBytes(MAX_BODY_BYTES + 1);
            if (bytes.length > MAX_BODY_BYTES) {
                throw new BadRequestException("Request body too large");
            }
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }

    /** Serialises {@code body} to JSON and sends it with the given status. */
    static void sendJson(HttpExchange exchange, int status, Object body) throws IOException {
        byte[] bytes = Json.toJson(body).getBytes(StandardCharsets.UTF_8);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        headers.set("Cache-Control", "no-store");
        send(exchange, status, bytes);
    }

    /** Sends {@code {"error": message}}. */
    static void sendError(HttpExchange exchange, int status, String message) throws IOException {
        sendJson(exchange, status, Map.of("error", message == null ? "Unknown error" : message));
    }

    /** Sends raw bytes; HEAD requests get headers only. */
    static void send(HttpExchange exchange, int status, byte[] bytes) throws IOException {
        boolean head = "HEAD".equalsIgnoreCase(exchange.getRequestMethod());
        if (head || bytes.length == 0) {
            exchange.sendResponseHeaders(status, -1);
            exchange.close();
            return;
        }
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }
}
