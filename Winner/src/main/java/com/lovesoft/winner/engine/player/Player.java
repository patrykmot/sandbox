package com.lovesoft.winner.engine.player;

import com.github.bhlangonijr.chesslib.Side;
import com.lovesoft.winner.engine.GameSnapshot;
import com.lovesoft.winner.engine.MoveRequest;

import java.util.Objects;
import java.util.Optional;

/**
 * A seat at the board.
 * <p>
 * The contract is {@link #makeMove(GameSnapshot)}: when it is this player's turn the game asks the player for a move.
 * <ul>
 *     <li>An <b>automatic</b> player (e.g. a future {@code BotPlayer}) answers immediately with a move.</li>
 *     <li>An <b>interactive</b> player ({@link HumanPlayer}) answers {@link Optional#empty()}, meaning
 *     "wait for my move to arrive from outside" (HTTP request, PGN replayer, ...).</li>
 * </ul>
 * This keeps the game loop identical for humans and bots, so adding a bot needs no change in the REST layer.
 */
public abstract class Player {

    private final String name;
    private final Side side;

    protected Player(String name, Side side) {
        this.name = Objects.requireNonNull(name, "name");
        this.side = Objects.requireNonNull(side, "side");
    }

    /** @return display name, e.g. "Player 1" */
    public String getName() {
        return name;
    }

    /** @return the colour this player controls */
    public Side getSide() {
        return side;
    }

    /**
     * Asks the player for a move in the given position.
     *
     * @param snapshot immutable view of the current game, with this player to move
     * @return the chosen move, or empty if the move will be submitted externally later
     */
    public abstract Optional<MoveRequest> makeMove(GameSnapshot snapshot);

    /**
     * @return {@code true} if moves for this player may be submitted from outside the engine
     * (e.g. via {@code POST /api/game/{id}/move})
     */
    public abstract boolean acceptsExternalMoves();

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{name='" + name + "', side=" + side + '}';
    }
}
