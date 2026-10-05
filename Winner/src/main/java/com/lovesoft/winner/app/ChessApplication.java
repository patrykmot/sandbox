package com.lovesoft.winner.app;

import com.lovesoft.winner.engine.GameManager;
import com.lovesoft.winner.web.ChessHttpServer;

import java.awt.Desktop;
import java.awt.GraphicsEnvironment;
import java.net.URI;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Entry point.
 * <ol>
 *     <li>creates the {@link GameManager} (the engine),</li>
 *     <li>starts the HTTP server (REST API + static frontend),</li>
 *     <li>opens the default browser at {@code http://localhost:<port>}.</li>
 * </ol>
 *
 * Usage: {@code java -jar chess-web-1.0.0.jar [port] [--no-browser]}
 * <br>The port can also be set with {@code -Dchess.port=9090}; default is 8080.
 * <br>The bind address can be set with {@code -Dchess.host=127.0.0.1}; default is 0.0.0.0 (all interfaces).
 */
public final class ChessApplication {

    private static final Logger LOG = Logger.getLogger(ChessApplication.class.getName());

    static final int DEFAULT_PORT = 8080;
    static final String DEFAULT_HOST = "0.0.0.0";

    private ChessApplication() {
    }

    public static void main(String[] args) throws Exception {
        int port = Integer.getInteger("chess.port", DEFAULT_PORT);
        String host = System.getProperty("chess.host", DEFAULT_HOST);
        boolean openBrowser = true;

        for (String arg : args) {
            if ("--no-browser".equalsIgnoreCase(arg)) {
                openBrowser = false;
            } else if (arg.matches("\\d{1,5}")) {
                port = Integer.parseInt(arg);
            } else {
                System.err.println("Ignoring unknown argument: " + arg);
            }
        }

        // 1. Engine
        GameManager gameManager = new GameManager();

        // 2. HTTP server
        ChessHttpServer server = new ChessHttpServer(host, port, gameManager);
        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> server.stop(1), "http-shutdown"));

        String url = "http://localhost:" + server.getPort();
        System.out.println("Chess server running at " + url + "  (Ctrl+C to stop)");

        // 3. Browser
        if (openBrowser) {
            openInBrowser(url);
        }
    }

    /** Opens the URL in the default browser when a desktop is available; otherwise just logs it. */
    static void openInBrowser(String url) {
        try {
            if (!GraphicsEnvironment.isHeadless()
                    && Desktop.isDesktopSupported()
                    && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                Desktop.getDesktop().browse(URI.create(url));
                return;
            }
            LOG.info(() -> "No desktop browser available; open " + url + " manually");
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Could not open the browser; open " + url + " manually", e);
        }
    }
}
