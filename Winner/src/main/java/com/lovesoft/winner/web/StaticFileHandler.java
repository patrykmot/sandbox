package com.lovesoft.winner.web;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Serves the frontend from the classpath folder {@code /web/} (i.e. {@code src/main/resources/web}).
 * {@code /} maps to {@code index.html}. Works the same from an IDE and from the fat JAR.
 */
public class StaticFileHandler implements HttpHandler {

    private static final Logger LOG = Logger.getLogger(StaticFileHandler.class.getName());

    private static final Map<String, String> CONTENT_TYPES = Map.of(
            "html", "text/html; charset=utf-8",
            "js", "application/javascript; charset=utf-8",
            "css", "text/css; charset=utf-8",
            "json", "application/json; charset=utf-8",
            "png", "image/png",
            "svg", "image/svg+xml",
            "ico", "image/x-icon");

    private final String classpathRoot;

    /** @param classpathRoot classpath folder holding the web files, e.g. {@code "/web"} */
    public StaticFileHandler(String classpathRoot) {
        this.classpathRoot = classpathRoot.endsWith("/")
                ? classpathRoot.substring(0, classpathRoot.length() - 1)
                : classpathRoot;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        try {
            String method = exchange.getRequestMethod().toUpperCase(Locale.ROOT);
            if (!"GET".equals(method) && !"HEAD".equals(method)) {
                exchange.getResponseHeaders().set("Allow", "GET, HEAD");
                HttpSupport.send(exchange, 405, new byte[0]);
                return;
            }

            String path = exchange.getRequestURI().getPath();
            if (path == null || path.isEmpty() || "/".equals(path)) {
                path = "/index.html";
            }
            // Reject traversal attempts and odd paths outright.
            if (path.contains("..") || path.contains("\\") || path.contains("//")) {
                HttpSupport.send(exchange, 400, new byte[0]);
                return;
            }

            byte[] bytes;
            try (InputStream in = StaticFileHandler.class.getResourceAsStream(classpathRoot + path)) {
                if (in == null) {
                    exchange.getResponseHeaders().set("Content-Type", "text/plain; charset=utf-8");
                    HttpSupport.send(exchange, 404, ("Not found: " + path).getBytes());
                    return;
                }
                bytes = in.readAllBytes();
            }

            exchange.getResponseHeaders().set("Content-Type", contentType(path));
            exchange.getResponseHeaders().set("Cache-Control", "no-cache");
            HttpSupport.send(exchange, 200, bytes);
        } catch (IOException e) {
            LOG.log(Level.FINE, "Client disconnected while serving " + exchange.getRequestURI(), e);
        } finally {
            exchange.close();
        }
    }

    private static String contentType(String path) {
        int dot = path.lastIndexOf('.');
        String ext = dot < 0 ? "" : path.substring(dot + 1).toLowerCase(Locale.ROOT);
        return CONTENT_TYPES.getOrDefault(ext, "application/octet-stream");
    }
}
