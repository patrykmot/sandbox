package com.lovesoft.winner.engine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Piece;
import com.github.bhlangonijr.chesslib.PieceType;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.Square;
import com.github.bhlangonijr.chesslib.move.Move;
import com.github.bhlangonijr.chesslib.move.MoveList;
import com.lovesoft.winner.engine.player.Player;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.regex.Pattern;

/**
 * One chess game: a chesslib {@link Board}, two {@link Player}s, the move history and the state machine.
 * <p>
 * <b>Thread safety:</b> chesslib's {@code Board} is not thread-safe, so every method that reads or mutates
 * game state is {@code synchronized} on this instance. Different games never share a lock, so they run in parallel.
 * <p>
 * <b>Isolation:</b> this class knows nothing about HTTP or JSON. It can be driven by the REST layer,
 * by unit tests, or by a headless PGN replayer in exactly the same way.
 */
public class Game {

    /** Safety net for bot-vs-bot loops; a real game is far shorter. */
    private static final int MAX_AUTOMATIC_PLIES = 2_000;

    private static final Pattern SQUARE = Pattern.compile("^[A-H][1-8]$");

    private final UUID id;
    private final Player white;
    private final Player black;
    private final Instant createdAt = Instant.now();

    private final Board board = new Board();
    private final List<String> sanHistory = new ArrayList<>();

    private GameStatus status = GameStatus.WAITING_TO_START;
    private ResultReason resultReason;
    private GameSnapshot.LastMove lastMove;

    public Game(UUID id, Player white, Player black) {
        this.id = Objects.requireNonNull(id, "id");
        this.white = Objects.requireNonNull(white, "white");
        this.black = Objects.requireNonNull(black, "black");
        if (white.getSide() != Side.WHITE || black.getSide() != Side.BLACK) {
            throw new IllegalArgumentException("White player must play WHITE and black player must play BLACK");
        }
    }

    // ------------------------------------------------------------------ state machine

    /**
     * WAITING_TO_START -> IN_PROGRESS. If the first player to move is automatic, it moves right away.
     *
     * @throws IllegalGameStateException if the game was already started
     */
    public synchronized GameSnapshot start() {
        if (status != GameStatus.WAITING_TO_START) {
            throw new IllegalGameStateException("Game " + id + " has already been started (status " + status + ")");
        }
        status = GameStatus.IN_PROGRESS;
        runAutomaticPlayers();
        return snapshot();
    }

    /**
     * Applies a move submitted from outside the engine (UI, API client, PGN replayer) for the side to move,
     * then lets any automatic player answer.
     *
     * @param request the move, in coordinate or SAN form
     * @return the state after the move (and after any automatic replies)
     * @throws IllegalGameStateException if the game is not in progress, or the side to move is not interactive
     * @throws IllegalMoveException      if the move is malformed or illegal; the board is unchanged
     */
    public synchronized GameSnapshot submitMove(MoveRequest request) {
        requireInProgress();
        Player toMove = currentPlayer();
        if (!toMove.acceptsExternalMoves()) {
            throw new IllegalGameStateException("It is " + toMove.getName() + "'s turn; external moves are not accepted");
        }
        applyMove(request);
        runAutomaticPlayers();
        return snapshot();
    }

    /** @return an immutable view of the current state */
    public synchronized GameSnapshot snapshot() {
        return new GameSnapshot(
                id.toString(),
                board.getFen(),
                board.getSideToMove().name(),
                status,
                white.getName(),
                black.getName(),
                lastMove,
                sanHistory,
                board.isKingAttacked(), // also true in a checkmate position
                resultReason);
    }

    // ------------------------------------------------------------------ accessors

    public UUID getId() {
        return id;
    }

    public Player getWhite() {
        return white;
    }

    public Player getBlack() {
        return black;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public synchronized GameStatus getStatus() {
        return status;
    }

    /** @return the player whose turn it is */
    public synchronized Player currentPlayer() {
        return board.getSideToMove() == Side.WHITE ? white : black;
    }

    // ------------------------------------------------------------------ internals (caller holds the lock)

    private void requireInProgress() {
        if (status != GameStatus.IN_PROGRESS) {
            throw new IllegalGameStateException("Game is not in progress (status " + status + ")");
        }
    }

    /** Lets automatic players (bots) move until an interactive player is to move or the game ends. */
    private void runAutomaticPlayers() {
        int plies = 0;
        while (status == GameStatus.IN_PROGRESS) {
            Optional<MoveRequest> reply = currentPlayer().makeMove(snapshot());
            if (reply.isEmpty()) {
                return; // interactive player: wait for an external move
            }
            applyMove(reply.get());
            if (++plies > MAX_AUTOMATIC_PLIES) {
                throw new IllegalStateException("Automatic players exceeded " + MAX_AUTOMATIC_PLIES + " plies");
            }
        }
    }

    /** Validates and plays one move, records it and updates the state machine. */
    private void applyMove(MoveRequest request) {
        Move move = resolveLegalMove(request);
        String san = toSan(move);
        if (!board.doMove(move, true)) {
            // Should not happen for a move taken from legalMoves(), but never leave the game inconsistent.
            throw new IllegalMoveException("Move " + move + " was rejected by the engine");
        }
        sanHistory.add(san);
        lastMove = new GameSnapshot.LastMove(
                move.getFrom().value().toLowerCase(Locale.ROOT),
                move.getTo().value().toLowerCase(Locale.ROOT),
                san);
        updateStatusAfterMove();
    }

    /**
     * Turns a {@link MoveRequest} into one of chesslib's legal moves for the current position.
     * Only moves contained in {@code board.legalMoves()} are ever returned.
     */
    private Move resolveLegalMove(MoveRequest request) {
        if (request == null) {
            throw new IllegalMoveException("A move is required");
        }
        List<Move> legalMoves = board.legalMoves();

        if (request.isSan()) {
            Move parsed = parseSan(request.san().trim());
            return legalMoves.stream()
                    .filter(parsed::equals)
                    .findFirst()
                    .orElseThrow(() -> new IllegalMoveException("Illegal move: " + request.san()));
        }

        if (request.isCoordinate()) {
            Square from = parseSquare(request.from());
            Square to = parseSquare(request.to());
            List<Move> candidates = legalMoves.stream()
                    .filter(m -> m.getFrom() == from && m.getTo() == to)
                    .toList();
            if (candidates.isEmpty()) {
                throw new IllegalMoveException("Illegal move: " + request.from() + "-" + request.to());
            }
            boolean isPromotion = candidates.get(0).getPromotion() != Piece.NONE;
            if (!isPromotion) {
                return candidates.get(0);
            }
            Piece promotion = promotionPiece(request.promotion(), board.getSideToMove());
            return candidates.stream()
                    .filter(m -> m.getPromotion() == promotion)
                    .findFirst()
                    .orElseThrow(() -> new IllegalMoveException("Illegal promotion piece: " + request.promotion()));
        }

        throw new IllegalMoveException("Provide either 'san', or both 'from' and 'to'");
    }

    /** Parses SAN in the context of the current position, using chesslib's SAN decoder. */
    private Move parseSan(String san) {
        try {
            MoveList moveList = new MoveList(board.getFen());
            moveList.addSanMove(san, true, true);
            return moveList.getLast();
        } catch (RuntimeException e) { // includes chesslib's MoveConversionException
            throw new IllegalMoveException("Illegal or unreadable SAN move: " + san, e);
        }
    }

    /** Encodes a legal move as SAN (with +/# suffixes) in the context of the current position. */
    private String toSan(Move move) {
        try {
            MoveList moveList = new MoveList(board.getFen());
            moveList.add(move);
            return moveList.toSanArray()[0];
        } catch (RuntimeException e) { // includes chesslib's MoveConversionException
            // SAN is cosmetic; fall back to coordinate notation rather than rejecting a legal move.
            return move.toString();
        }
    }

    private static Square parseSquare(String value) {
        String normalized = value == null ? "" : value.trim().toUpperCase(Locale.ROOT);
        if (!SQUARE.matcher(normalized).matches()) {
            throw new IllegalMoveException("Invalid square: " + value);
        }
        return Square.valueOf(normalized);
    }

    /** Maps "q"/"r"/"b"/"n" (any case) to a piece of the given side; a missing value means queen. */
    private static Piece promotionPiece(String letter, Side side) {
        String normalized = letter == null || letter.isBlank() ? "q" : letter.trim().toLowerCase(Locale.ROOT);
        PieceType type = switch (normalized) {
            case "q" -> PieceType.QUEEN;
            case "r" -> PieceType.ROOK;
            case "b" -> PieceType.BISHOP;
            case "n" -> PieceType.KNIGHT;
            default -> throw new IllegalMoveException("Invalid promotion piece: " + letter + " (use q, r, b or n)");
        };
        return Piece.make(side, type);
    }

    /** Checks for mate and every draw rule after a move. */
    private void updateStatusAfterMove() {
        if (board.isMated()) {
            // The side to move is mated, so the side that just moved wins.
            status = board.getSideToMove() == Side.WHITE ? GameStatus.FINISHED_BLACK_WON : GameStatus.FINISHED_WHITE_WON;
            resultReason = ResultReason.CHECKMATE;
        } else if (board.isStaleMate()) {
            finishAsDraw(ResultReason.STALEMATE);
        } else if (board.isInsufficientMaterial()) {
            finishAsDraw(ResultReason.INSUFFICIENT_MATERIAL);
        } else if (board.isRepetition()) {
            finishAsDraw(ResultReason.THREEFOLD_REPETITION);
        } else if (board.getHalfMoveCounter() >= 100) {
            finishAsDraw(ResultReason.FIFTY_MOVE_RULE);
        }
    }

    private void finishAsDraw(ResultReason reason) {
        status = GameStatus.FINISHED_DRAW;
        resultReason = reason;
    }

    /**
     * Test/replay hook: loads a starting position before the game starts.
     *
     * @param fen position in FEN
     * @throws IllegalGameStateException if the game has already started
     */
    public synchronized void loadPosition(String fen) {
        if (status != GameStatus.WAITING_TO_START) {
            throw new IllegalGameStateException("Position can only be set before the game starts");
        }
        board.loadFromFen(fen);
    }
}
