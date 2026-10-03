package com.lovesoft.winner.engine;

/**
 * Transport-neutral description of a move, as submitted by a UI, a bot or a PGN replayer.
 * <p>
 * Exactly one form must be used:
 * <ul>
 *     <li>coordinate form: {@code from} + {@code to} (e.g. "e2", "e4") and an optional {@code promotion}
 *     piece letter ("q", "r", "b", "n"); a missing promotion on a promoting move defaults to a queen;</li>
 *     <li>SAN form: {@code san} (e.g. "Nf3", "exd5", "O-O", "e8=Q+").</li>
 * </ul>
 *
 * @param from      origin square, case-insensitive (coordinate form)
 * @param to        target square, case-insensitive (coordinate form)
 * @param promotion optional promotion piece letter (coordinate form)
 * @param san       move in Standard Algebraic Notation (SAN form)
 */
public record MoveRequest(String from, String to, String promotion, String san) {

    /** Coordinate move without promotion. */
    public static MoveRequest of(String from, String to) {
        return new MoveRequest(from, to, null, null);
    }

    /** Coordinate move with an explicit promotion piece letter. */
    public static MoveRequest of(String from, String to, String promotion) {
        return new MoveRequest(from, to, promotion, null);
    }

    /** SAN move. */
    public static MoveRequest ofSan(String san) {
        return new MoveRequest(null, null, null, san);
    }

    /** @return {@code true} if this request uses the SAN form */
    public boolean isSan() {
        return san != null && !san.isBlank();
    }

    /** @return {@code true} if this request uses the coordinate form */
    public boolean isCoordinate() {
        return from != null && !from.isBlank() && to != null && !to.isBlank();
    }
}
