package com.lovesoft.winner.engine;

/** Why a game ended. Only set once the game is in a finished {@link GameStatus}. */
public enum ResultReason {
    CHECKMATE,
    STALEMATE,
    INSUFFICIENT_MATERIAL,
    THREEFOLD_REPETITION,
    FIFTY_MOVE_RULE
}
