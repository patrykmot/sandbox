package com.lovesoft.winner.engine.player;

import com.github.bhlangonijr.chesslib.Side;
import com.lovesoft.winner.engine.GameSnapshot;
import com.lovesoft.winner.engine.MoveRequest;

import java.util.Optional;

/**
 * A human seat. Humans never answer synchronously: their moves arrive later through the API
 * (or from a PGN replayer acting on their behalf).
 */
public class HumanPlayer extends Player {

    public HumanPlayer(String name, Side side) {
        super(name, side);
    }

    /** Always empty: the game waits for an external move. */
    @Override
    public Optional<MoveRequest> makeMove(GameSnapshot snapshot) {
        return Optional.empty();
    }

    @Override
    public boolean acceptsExternalMoves() {
        return true;
    }
}
