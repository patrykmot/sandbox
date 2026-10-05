package com.lovesoft.winner.web;

import com.lovesoft.winner.engine.Game;
import com.lovesoft.winner.engine.GameManager;
import com.lovesoft.winner.engine.GameSnapshot;
import com.lovesoft.winner.engine.IllegalGameStateException;
import com.lovesoft.winner.engine.IllegalMoveException;
import com.lovesoft.winner.engine.MoveRequest;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * REST API for games, mounted at {@code /api/game}.
 *
 * <pre>
 * POST /api/game/start       body (optional): {"white":"Alice","black":"Bob"}        -> 201 {gameId, state}
 * GET  /api/game/{id}/state                                                         -> 200 state
 * POST /api/game/{id}/move   body: {"from":"e2","to":"e4","promotion":"q"} | {"san":"Nf3"} -> 200 state
 * </pre>
 *
 * Errors are {@code {"error": "..."}} with 400 (bad input), 404 (unknown game/route), 405 (wrong method),
 * 409 (game not in progress) or 422 (illegal move).
 */
public class GameApiHandler implements HttpHandler {

    private static final Logger LOG = Logger.getLogger(GameApiHandler.class.getName());

    private static final String START_PATH = "/api/game/start";
    private static final Pattern GAME_PATH = Pattern.compile("^/api/game/([^/]+)/(state|move)$");
    private static final Pattern UUID_PATTERN = Pattern.compile(
            "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$");

    private final GameManager gameManager;

    public GameApiHandler(GameManager gameManager) {
        this.gameManager = Objects.requireNonNull(gameManager, "gameManager");
    }

    /** Body of {@code POST /api/game/start}; both fields optional. */
    record StartRequest(String white, String black) {
    }

    /** Response of {@code POST /api/game/start}. */
    record StartResponse(String gameId, GameSnapshot state) {
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        try {
            route(exchange);
        } catch (BadRequestException e) {
            HttpSupport.sendError(exchange, 400, e.getMessage());
        } catch (NotFoundException e) {
            HttpSupport.sendError(exchange, 404, e.getMessage());
        } catch (MethodNotAllowedException e) {
            exchange.getResponseHeaders().set("Allow", e.allowed);
            HttpSupport.sendError(exchange, 405, "Method not allowed; use " + e.allowed);
        } catch (IllegalGameStateException e) {
            HttpSupport.sendError(exchange, 409, e.getMessage());
        } catch (IllegalMoveException e) {
            HttpSupport.sendError(exchange, 422, e.getMessage());
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Unexpected error for " + exchange.getRequestMethod() + " "
                    + exchange.getRequestURI(), e);
            HttpSupport.sendError(exchange, 500, "Internal server error");
        } finally {
            exchange.close();
        }
    }

    private void route(HttpExchange exchange) throws IOException {
        String path = stripTrailingSlash(exchange.getRequestURI().getPath());
        String method = exchange.getRequestMethod().toUpperCase();

        if (START_PATH.equals(path)) {
            requireMethod(method, "POST");
            handleStart(exchange);
            return;
        }

        Matcher matcher = GAME_PATH.matcher(path);
        if (!matcher.matches()) {
            throw new NotFoundException("Unknown API path: " + path);
        }
        Game game = findGame(matcher.group(1));
        if ("state".equals(matcher.group(2))) {
            requireMethod(method, "GET");
            HttpSupport.sendJson(exchange, 200, game.snapshot());
        } else {
            requireMethod(method, "POST");
            handleMove(exchange, game);
        }
    }

    private void handleStart(HttpExchange exchange) throws IOException {
        StartRequest request = Json.fromJson(HttpSupport.readBody(exchange), StartRequest.class);
        String white = request == null ? null : request.white();
        String black = request == null ? null : request.black();
        Game game = gameManager.createHumanGame(white, black);
        LOG.info(() -> "Started game " + game.getId());
        HttpSupport.sendJson(exchange, 201, new StartResponse(game.getId().toString(), game.snapshot()));
    }

    private void handleMove(HttpExchange exchange, Game game) throws IOException {
        MoveRequest move = Json.fromJson(HttpSupport.readBody(exchange), MoveRequest.class);
        if (move == null) {
            throw new BadRequestException("Request body with a move is required");
        }
        GameSnapshot state = game.submitMove(move); // throws IllegalMove / IllegalGameState
        HttpSupport.sendJson(exchange, 200, state);
    }

    private Game findGame(String id) {
        if (!UUID_PATTERN.matcher(id).matches()) {
            throw new BadRequestException("Malformed game id: " + id);
        }
        return gameManager.getGame(id).orElseThrow(() -> new NotFoundException("Unknown game: " + id));
    }

    private static void requireMethod(String actual, String expected) {
        if (!expected.equals(actual)) {
            throw new MethodNotAllowedException(expected);
        }
    }

    private static String stripTrailingSlash(String path) {
        return path.length() > 1 && path.endsWith("/") ? path.substring(0, path.length() - 1) : path;
    }

    private static final class NotFoundException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        NotFoundException(String message) {
            super(message);
        }
    }

    private static final class MethodNotAllowedException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        final String allowed;

        MethodNotAllowedException(String allowed) {
            super(allowed);
            this.allowed = allowed;
        }
    }
}
