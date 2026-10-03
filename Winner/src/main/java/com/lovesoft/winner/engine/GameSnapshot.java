package com.lovesoft.winner.engine;

import java.util.List;

/**
 * Immutable, point-in-time view of a game. This is the only game data handed out of the engine,
 * so callers (REST layer, bots, PGN replayer) can never mutate the live board.
 *
 * @param gameId       game UUID as a string
 * @param fen          current position in Forsyth-Edwards Notation
 * @param turn         side to move: "WHITE" or "BLACK"
 * @param status       state-machine status
 * @param whitePlayer  white player's display name
 * @param blackPlayer  black player's display name
 * @param lastMove     last move played, or {@code null} before the first move
 * @param moveHistory  all moves so far in SAN, in order
 * @param inCheck      whether the side to move is in check (also true when it is checkmated)
 * @param resultReason why the game ended, or {@code null} while it is not finished
 */
public record GameSnapshot(
        String gameId,
        String fen,
        String turn,
        GameStatus status,
        String whitePlayer,
        String blackPlayer,
        LastMove lastMove,
        List<String> moveHistory,
        boolean inCheck,
        ResultReason resultReason) {

    public GameSnapshot {
        moveHistory = List.copyOf(moveHistory);
    }

    /**
     * The most recent move.
     *
     * @param from origin square, lower case (e.g. "e2")
     * @param to   target square, lower case (e.g. "e4")
     * @param san  the move in SAN (e.g. "e4")
     */
    public record LastMove(String from, String to, String san) {
    }
}
