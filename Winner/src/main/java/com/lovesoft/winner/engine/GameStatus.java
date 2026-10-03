package com.lovesoft.winner.engine;

/**
 * Lifecycle of a single game.
 *
 * <pre>
 * WAITING_TO_START --start()--> IN_PROGRESS --mate/draw--> FINISHED_WHITE_WON | FINISHED_BLACK_WON | FINISHED_DRAW
 * </pre>
 *
 * Finished states are terminal: no further moves are accepted.
 */
public enum GameStatus {
    WAITING_TO_START,
    IN_PROGRESS,
    FINISHED_WHITE_WON,
    FINISHED_BLACK_WON,
    FINISHED_DRAW;

    /** @return {@code true} for the three terminal states */
    public boolean isFinished() {
        return this == FINISHED_WHITE_WON || this == FINISHED_BLACK_WON || this == FINISHED_DRAW;
    }
}
