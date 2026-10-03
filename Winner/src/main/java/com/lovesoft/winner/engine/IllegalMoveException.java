package com.lovesoft.winner.engine;

/** Thrown when a submitted move is malformed or not legal in the current position. The board is left untouched. */
public class IllegalMoveException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public IllegalMoveException(String message) {
        super(message);
    }

    public IllegalMoveException(String message, Throwable cause) {
        super(message, cause);
    }
}
