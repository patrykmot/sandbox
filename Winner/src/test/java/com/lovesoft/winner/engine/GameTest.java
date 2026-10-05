package com.lovesoft.winner.engine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import com.lovesoft.winner.engine.player.HumanPlayer;
import com.lovesoft.winner.engine.player.Player;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GameTest {

    private static final String START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    private Game game;

    @BeforeEach
    void setUp() {
        game = newHumanGame();
    }

    private static Game newHumanGame() {
        return new Game(UUID.randomUUID(),
                new HumanPlayer("Player 1", Side.WHITE),
                new HumanPlayer("Player 2", Side.BLACK));
    }

    @Test
    void newGameWaitsUntilStarted() {
        GameSnapshot s = game.snapshot();
        assertEquals(GameStatus.WAITING_TO_START, s.status());
        assertEquals(START_FEN, s.fen());
        assertThrows(IllegalGameStateException.class, () -> game.submitMove(MoveRequest.of("e2", "e4")));

        assertEquals(GameStatus.IN_PROGRESS, game.start().status());
        assertThrows(IllegalGameStateException.class, game::start);
    }

    @Test
    void legalCoordinateMoveUpdatesFenTurnAndHistory() {
        game.start();
        GameSnapshot s = game.submitMove(MoveRequest.of("e2", "e4"));

        assertEquals("BLACK", s.turn());
        assertTrue(s.fen().startsWith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq"));
        assertEquals(List.of("e4"), s.moveHistory());
        assertEquals(new GameSnapshot.LastMove("e2", "e4", "e4"), s.lastMove());
        assertFalse(s.inCheck());
        assertNull(s.resultReason());
    }

    @Test
    void squaresAreCaseInsensitive() {
        game.start();
        assertEquals("BLACK", game.submitMove(MoveRequest.of("G1", "F3")).turn());
    }

    @Test
    void illegalMoveIsRejectedAndBoardIsUnchanged() {
        game.start();
        String before = game.snapshot().fen();

        assertThrows(IllegalMoveException.class, () -> game.submitMove(MoveRequest.of("e2", "e5")));
        assertThrows(IllegalMoveException.class, () -> game.submitMove(MoveRequest.of("e7", "e5"))); // not white's piece
        assertThrows(IllegalMoveException.class, () -> game.submitMove(MoveRequest.of("z9", "e4")));
        assertThrows(IllegalMoveException.class, () -> game.submitMove(MoveRequest.ofSan("Qh5")));
        assertThrows(IllegalMoveException.class, () -> game.submitMove(MoveRequest.ofSan("nonsense")));
        assertThrows(IllegalMoveException.class, () -> game.submitMove(new MoveRequest(null, null, null, null)));
        assertThrows(IllegalMoveException.class, () -> game.submitMove(null));

        assertEquals(before, game.snapshot().fen());
        assertTrue(game.snapshot().moveHistory().isEmpty());
    }

    @Test
    void sanAndCoordinateInputGiveTheSamePosition() {
        Game viaSan = newHumanGame();
        viaSan.start();
        viaSan.submitMove(MoveRequest.ofSan("e4"));
        viaSan.submitMove(MoveRequest.ofSan("e5"));
        GameSnapshot san = viaSan.submitMove(MoveRequest.ofSan("Nf3"));

        game.start();
        game.submitMove(MoveRequest.of("e2", "e4"));
        game.submitMove(MoveRequest.of("e7", "e5"));
        GameSnapshot coordinate = game.submitMove(MoveRequest.of("g1", "f3"));

        assertEquals(san.fen(), coordinate.fen());
        assertEquals(List.of("e4", "e5", "Nf3"), coordinate.moveHistory());
    }

    @Test
    void foolsMateEndsWithBlackWinning() {
        game.start();
        game.submitMove(MoveRequest.ofSan("f3"));
        game.submitMove(MoveRequest.ofSan("e5"));
        game.submitMove(MoveRequest.ofSan("g4"));
        GameSnapshot s = game.submitMove(MoveRequest.of("d8", "h4"));

        assertEquals(GameStatus.FINISHED_BLACK_WON, s.status());
        assertEquals(ResultReason.CHECKMATE, s.resultReason());
        assertEquals("Qh4#", s.lastMove().san());
        assertTrue(s.status().isFinished());
        assertThrows(IllegalGameStateException.class, () -> game.submitMove(MoveRequest.ofSan("a3")));
    }

    @Test
    void scholarsMateEndsWithWhiteWinning() {
        game.start();
        for (String san : List.of("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6")) {
            game.submitMove(MoveRequest.ofSan(san));
        }
        GameSnapshot s = game.submitMove(MoveRequest.ofSan("Qxf7#"));
        assertEquals(GameStatus.FINISHED_WHITE_WON, s.status());
        assertEquals(ResultReason.CHECKMATE, s.resultReason());
    }

    @Test
    void stalemateEndsInDraw() {
        game.loadPosition("7k/8/6K1/5Q2/8/8/8/8 w - - 0 1");
        game.start();
        GameSnapshot s = game.submitMove(MoveRequest.of("f5", "f7"));

        assertEquals(GameStatus.FINISHED_DRAW, s.status());
        assertEquals(ResultReason.STALEMATE, s.resultReason());
    }

    @Test
    void insufficientMaterialEndsInDraw() {
        game.loadPosition("8/8/8/4k3/8/8/3Kn3/8 w - - 0 1");
        game.start();
        GameSnapshot s = game.submitMove(MoveRequest.of("d2", "e2"));

        assertEquals(GameStatus.FINISHED_DRAW, s.status());
        assertEquals(ResultReason.INSUFFICIENT_MATERIAL, s.resultReason());
    }

    @Test
    void threefoldRepetitionEndsInDraw() {
        game.start();
        GameSnapshot s = null;
        for (int i = 0; i < 2; i++) {
            game.submitMove(MoveRequest.ofSan("Nf3"));
            game.submitMove(MoveRequest.ofSan("Nf6"));
            game.submitMove(MoveRequest.ofSan("Ng1"));
            s = game.submitMove(MoveRequest.ofSan("Ng8"));
        }
        assertEquals(GameStatus.FINISHED_DRAW, s.status());
        assertEquals(ResultReason.THREEFOLD_REPETITION, s.resultReason());
    }

    @Test
    void promotionDefaultsToQueenAndAcceptsUnderPromotion() {
        game.loadPosition("8/P7/8/8/8/8/k7/4K3 w - - 0 1");
        game.start();
        GameSnapshot queen = game.submitMove(MoveRequest.of("a7", "a8"));
        assertTrue(queen.fen().startsWith("Q7/"), queen.fen());
        assertEquals("a8=Q+", queen.lastMove().san());

        Game other = newHumanGame();
        other.loadPosition("8/P7/8/8/8/8/k7/4K3 w - - 0 1");
        other.start();
        GameSnapshot knight = other.submitMove(MoveRequest.of("a7", "a8", "n"));
        assertTrue(knight.fen().startsWith("N7/"), knight.fen());

        Game bad = newHumanGame();
        bad.loadPosition("8/P7/8/8/8/8/k7/4K3 w - - 0 1");
        bad.start();
        assertThrows(IllegalMoveException.class, () -> bad.submitMove(MoveRequest.of("a7", "a8", "k")));
    }

    @Test
    void castlingWithCoordinatesMovesTheRookToo() {
        game.start();
        for (String san : List.of("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")) {
            game.submitMove(MoveRequest.ofSan(san));
        }
        GameSnapshot s = game.submitMove(MoveRequest.of("e1", "g1"));
        assertEquals("O-O", s.lastMove().san());
        assertTrue(s.fen().contains("RNBQ1RK1"), s.fen());
    }

    @Test
    void automaticPlayerRepliesImmediately() {
        Game vsBot = new Game(UUID.randomUUID(),
                new HumanPlayer("Human", Side.WHITE),
                new FirstLegalMoveBot("Bot", Side.BLACK));
        vsBot.start();
        GameSnapshot s = vsBot.submitMove(MoveRequest.ofSan("e4"));

        assertEquals(2, s.moveHistory().size());
        assertEquals("WHITE", s.turn());
    }

    @Test
    void automaticPlayerMovesFirstWhenItPlaysWhite() {
        Game botFirst = new Game(UUID.randomUUID(),
                new FirstLegalMoveBot("Bot", Side.WHITE),
                new HumanPlayer("Human", Side.BLACK));
        GameSnapshot s = botFirst.start(); // bot plays white's first move on start
        assertEquals(1, s.moveHistory().size());
        assertEquals("BLACK", s.turn());
    }

    /** Minimal automatic player used to exercise the Player contract (a stand-in for a future BotPlayer). */
    private static final class FirstLegalMoveBot extends Player {

        FirstLegalMoveBot(String name, Side side) {
            super(name, side);
        }

        @Override
        public Optional<MoveRequest> makeMove(GameSnapshot snapshot) {
            Board board = new Board();
            board.loadFromFen(snapshot.fen());
            Move move = board.legalMoves().get(0);
            return Optional.of(MoveRequest.of(
                    move.getFrom().value().toLowerCase(Locale.ROOT),
                    move.getTo().value().toLowerCase(Locale.ROOT)));
        }

        @Override
        public boolean acceptsExternalMoves() {
            return false;
        }
    }
}
