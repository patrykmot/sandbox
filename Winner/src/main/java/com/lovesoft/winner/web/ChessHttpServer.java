package com.lovesoft.winner.web;

import com.lovesoft.winner.engine.GameManager;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

/**
 * Wraps the JDK's built-in {@link HttpServer}: one context for the REST API, one for static files,
 * and a fixed worker pool so several browser tabs (and games) are served concurrently.
 */
public class ChessHttpServer {

    private static final Logger LOG = Logger.getLogger(ChessHttpServer.class.getName());
    private static final int WORKER_THREADS = 8;

    private final HttpServer server;
    private final ExecutorService executor;

    /**
     * @param bindHost    interface to bind, e.g. "0.0.0.0" (all) or "127.0.0.1" (local only)
     * @param port        TCP port; 0 picks a free port (handy for tests)
     * @param gameManager the shared game registry
     */
    public ChessHttpServer(String bindHost, int port, GameManager gameManager) throws IOException {
        Objects.requireNonNull(gameManager, "gameManager");
        this.server = HttpServer.create(new InetSocketAddress(bindHost, port), 0);

        AtomicInteger counter = new AtomicInteger();
        this.executor = Executors.newFixedThreadPool(WORKER_THREADS, runnable -> {
            Thread thread = new Thread(runnable, "http-worker-" + counter.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        });
        server.setExecutor(executor);

        // The most specific context wins, so /api/game/* goes to the API and everything else to static files.
        server.createContext("/api/game", new GameApiHandler(gameManager));
        server.createContext("/", new StaticFileHandler("/web"));
    }

    public void start() {
        server.start();
        LOG.info(() -> "HTTP server listening on port " + getPort());
    }

    /** Stops accepting requests, waits up to {@code delaySeconds} for running ones, then frees the threads. */
    public void stop(int delaySeconds) {
        server.stop(delaySeconds);
        executor.shutdown();
        try {
            if (!executor.awaitTermination(delaySeconds + 1L, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        LOG.info("HTTP server stopped");
    }

    /** @return the actual bound port (useful when constructed with port 0) */
    public int getPort() {
        return server.getAddress().getPort();
    }
}
