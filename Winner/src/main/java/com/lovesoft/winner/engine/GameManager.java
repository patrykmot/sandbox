package com.lovesoft.winner.engine;

import com.github.bhlangonijr.chesslib.Side;
import com.lovesoft.winner.engine.player.HumanPlayer;
import com.lovesoft.winner.engine.player.Player;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of all running games, keyed by UUID. Safe to use from many HTTP worker threads at once:
 * the map is a {@link ConcurrentHashMap} and each {@link Game} synchronizes on itself.
 */
public class GameManager {

    public static final String DEFAULT_WHITE_NAME = "Player 1";
    public static final String DEFAULT_BLACK_NAME = "Player 2";
    private static final int MAX_NAME_LENGTH = 40;

    private final Map<UUID, Game> games = new ConcurrentHashMap<>();

    /**
     * Creates and starts a game between two human players.
     *
     * @param whiteName white's display name, or {@code null}/blank for "Player 1"
     * @param blackName black's display name, or {@code null}/blank for "Player 2"
     * @return the new game, already {@link GameStatus#IN_PROGRESS}
     */
    public Game createHumanGame(String whiteName, String blackName) {
        Player white = new HumanPlayer(cleanName(whiteName, DEFAULT_WHITE_NAME), Side.WHITE);
        Player black = new HumanPlayer(cleanName(blackName, DEFAULT_BLACK_NAME), Side.BLACK);
        Game game = createGame(white, black);
        game.start(); // both seats are filled, so the game can begin right away
        return game;
    }

    /**
     * Registers a game for any two players (e.g. a future BotPlayer) without starting it.
     * Call {@link Game#start()} when ready.
     */
    public Game createGame(Player white, Player black) {
        Game game = new Game(UUID.randomUUID(), white, black);
        games.put(game.getId(), game);
        return game;
    }

    /** @return the game with this id, if any */
    public Optional<Game> getGame(UUID id) {
        return Optional.ofNullable(games.get(id));
    }

    /**
     * Looks up a game by its string id.
     *
     * @throws IllegalArgumentException if {@code id} is not a valid UUID
     */
    public Optional<Game> getGame(String id) {
        return getGame(UUID.fromString(id));
    }

    /** @return a read-only view of all games */
    public Collection<Game> listGames() {
        return Collections.unmodifiableCollection(games.values());
    }

    /** @return {@code true} if a game was removed */
    public boolean removeGame(UUID id) {
        return games.remove(id) != null;
    }

    public int size() {
        return games.size();
    }

    private static String cleanName(String name, String fallback) {
        if (name == null || name.isBlank()) {
            return fallback;
        }
        String trimmed = name.strip();
        return trimmed.length() > MAX_NAME_LENGTH ? trimmed.substring(0, MAX_NAME_LENGTH) : trimmed;
    }
}
