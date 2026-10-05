package com.lovesoft.winner.engine;

/**
 * Thrown when an operation does not fit the game's current state, e.g. a move submitted to a finished game,
 * or an external move submitted while an automatic player (bot) is to move.
 */
public class IllegalGameStateException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public IllegalGameStateException(String message) {
        super(message);
    }
}
