/*
 * Winner Chess frontend.
 *
 * The page never decides whether a move is legal: every drop is POSTed to the backend, which validates it
 * with chesslib. The page only renders whatever state the backend returns, and polls for changes made in
 * other tabs.
 *
 * URL parameters:
 *   ?game=<uuid>   join an existing game (omitted -> a new game is created)
 *   &side=black    show the board from Black's side in this tab
 */
$(function () {
    'use strict';

    const POLL_INTERVAL_MS = 1000;
    const PIECE_THEME = 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png';

    const STATUS_VIEW = {
        WAITING_TO_START:   { text: 'Waiting to start', css: 'text-bg-secondary' },
        IN_PROGRESS:        { text: 'In progress',      css: 'text-bg-primary' },
        FINISHED_WHITE_WON: { text: 'White won',        css: 'text-bg-success' },
        FINISHED_BLACK_WON: { text: 'Black won',        css: 'text-bg-dark' },
        FINISHED_DRAW:      { text: 'Draw',             css: 'text-bg-warning' }
    };

    const REASON_TEXT = {
        CHECKMATE: 'by checkmate',
        STALEMATE: 'by stalemate',
        INSUFFICIENT_MATERIAL: 'insufficient material',
        THREEFOLD_REPETITION: 'threefold repetition',
        FIFTY_MOVE_RULE: '50-move rule'
    };

    // ---------------------------------------------------------------- state
    let gameId = null;
    let lastState = null;        // last state confirmed by the backend
    let moveInFlight = false;    // a POST /move is pending
    let dragging = false;        // the user is holding a piece
    let moveCounter = 0;         // bumps on every move request, to discard polls that raced with it

    const params = new URLSearchParams(window.location.search);
    const board = Chessboard('board', {
        position: 'start',
        draggable: true,
        pieceTheme: PIECE_THEME,
        orientation: params.get('side') === 'black' ? 'black' : 'white',
        onDragStart: onDragStart,
        onDrop: onDrop
    });
    const $board = $('#board');
    const errorToast = bootstrap.Toast.getOrCreateInstance(document.getElementById('error-toast'));

    // ---------------------------------------------------------------- API
    const api = {
        start: () => $.ajax({
            url: '/api/game/start', method: 'POST',
            contentType: 'application/json', data: '{}', dataType: 'json'
        }),
        state: (id) => $.ajax({ url: `/api/game/${id}/state`, method: 'GET', dataType: 'json', cache: false }),
        move: (id, move) => $.ajax({
            url: `/api/game/${id}/move`, method: 'POST',
            contentType: 'application/json', data: JSON.stringify(move), dataType: 'json'
        })
    };

    // ---------------------------------------------------------------- board callbacks

    /** Only allow dragging the side-to-move's pieces while the game is running and nothing is pending. */
    function onDragStart(source, piece) {
        if (!lastState || moveInFlight || lastState.status !== 'IN_PROGRESS') {
            return false;
        }
        const pieceSide = piece.charAt(0) === 'w' ? 'WHITE' : 'BLACK';
        if (pieceSide !== lastState.turn) {
            return false;
        }
        dragging = true;
        return true;
    }

    /**
     * chessboard.js wants a synchronous answer, but validation happens on the server. So the drop is
     * accepted visually; on success we render the server's FEN (which also fixes castling, en passant
     * and promotion), on error we restore the last confirmed position, i.e. the piece snaps back.
     */
    function onDrop(source, target) {
        dragging = false;
        if (target === 'offboard' || source === target) {
            return 'snapback';
        }

        moveInFlight = true;
        moveCounter++;
        // promotion is only used by the backend when the move actually promotes
        api.move(gameId, { from: source, to: target, promotion: 'q' })
            .done(render)
            .fail((xhr) => {
                board.position(lastState.fen, true);
                showError(errorMessage(xhr, 'Move rejected'));
            })
            .always(() => { moveInFlight = false; });
        return undefined;
    }

    // ---------------------------------------------------------------- rendering

    function render(state) {
        const previous = lastState;
        lastState = state;

        if (!previous || previous.fen !== state.fen) {
            board.position(state.fen, !!previous);
        }

        $('#game-id').text(state.gameId);
        $('#player-white .name').text(state.whitePlayer);
        $('#player-black .name').text(state.blackPlayer);

        const running = state.status === 'IN_PROGRESS';
        $('#player-white').toggleClass('to-move', running && state.turn === 'WHITE');
        $('#player-black').toggleClass('to-move', running && state.turn === 'BLACK');

        const view = STATUS_VIEW[state.status] || { text: state.status, css: 'text-bg-secondary' };
        $('#status-badge').attr('class', 'badge ' + view.css).text(view.text);
        $('#status-detail').text(statusDetail(state));

        renderMoveList(state.moveHistory);
        renderHighlights(state);
    }

    function statusDetail(state) {
        if (state.status === 'IN_PROGRESS') {
            const side = state.turn === 'WHITE' ? 'White' : 'Black';
            return state.inCheck ? `${side} to move — check!` : `${side} to move`;
        }
        return state.resultReason ? REASON_TEXT[state.resultReason] || state.resultReason : '';
    }

    function renderMoveList(moves) {
        const $list = $('#move-list').empty();
        $('#move-list-empty').toggle(moves.length === 0);
        for (let i = 0; i < moves.length; i += 2) {
            const $li = $('<li>');
            $li.append($('<span class="ply">').text(moves[i]));
            if (i + 1 < moves.length) {
                $li.append($('<span class="ply">').text(moves[i + 1]));
            }
            $list.append($li);
        }
        $list.find('.ply').last().addClass('last');
        $list.scrollTop($list.prop('scrollHeight'));
    }

    function renderHighlights(state) {
        $board.find('.highlight-last').removeClass('highlight-last');
        $board.find('.highlight-check').removeClass('highlight-check');

        if (state.lastMove) {
            $board.find('.square-' + state.lastMove.from).addClass('highlight-last');
            $board.find('.square-' + state.lastMove.to).addClass('highlight-last');
        }
        if (state.inCheck) {
            const king = state.turn === 'WHITE' ? 'wK' : 'bK';
            const position = board.position();
            const square = Object.keys(position).find((sq) => position[sq] === king);
            if (square) {
                $board.find('.square-' + square).addClass('highlight-check');
            }
        }
    }

    // ---------------------------------------------------------------- game lifecycle

    /** @param {boolean} pushHistory true when the user asked for a new game (keeps Back working) */
    function startNewGame(pushHistory) {
        return api.start()
            .done((response) => {
                setGameId(response.gameId, pushHistory === true);
                lastState = null;
                render(response.state);
            })
            .fail((xhr) => showError(errorMessage(xhr, 'Could not start a new game')));
    }

    function joinGame(id) {
        return api.state(id)
            .done((state) => {
                setGameId(id, false);
                render(state);
            })
            .fail((xhr) => {
                showError(errorMessage(xhr, 'Could not load game') + ' — starting a new game.');
                startNewGame(false);
            });
    }

    function setGameId(id, pushHistory) {
        gameId = id;
        const url = new URL(window.location.href);
        url.searchParams.set('game', id);
        if (pushHistory) {
            window.history.pushState({}, '', url);
        } else {
            window.history.replaceState({}, '', url);
        }
        $('#btn-copy').prop('disabled', false);
    }

    /** Polls the backend so moves made in other tabs show up here. */
    function poll() {
        if (!gameId || dragging || moveInFlight) {
            return;
        }
        const counterAtRequest = moveCounter;
        const idAtRequest = gameId;
        api.state(idAtRequest).done((state) => {
            // Drop responses that raced with a local move, a drag, or a game switch.
            if (dragging || moveInFlight || counterAtRequest !== moveCounter || idAtRequest !== gameId) {
                return;
            }
            if (!lastState || lastState.fen !== state.fen || lastState.status !== state.status) {
                render(state);
            }
        });
    }

    // ---------------------------------------------------------------- helpers

    function errorMessage(xhr, fallback) {
        return (xhr && xhr.responseJSON && xhr.responseJSON.error) || fallback;
    }

    function showError(message) {
        $('#error-toast-body').text(message);
        errorToast.show();
    }

    // ---------------------------------------------------------------- wiring

    $('#btn-new').on('click', () => startNewGame(true));
    $('#btn-flip').on('click', () => board.flip());
    $('#btn-copy').on('click', function () {
        const $btn = $(this);
        navigator.clipboard.writeText(window.location.href).then(() => {
            $btn.text('Copied!');
            setTimeout(() => $btn.text('Copy link'), 1500);
        });
    });
    $(window).on('resize', () => board.resize());
    window.addEventListener('popstate', () => {
        const id = new URLSearchParams(window.location.search).get('game');
        if (id && id !== gameId) {
            lastState = null;
            joinGame(id);
        }
    });

    const initialId = params.get('game');
    if (initialId) {
        joinGame(initialId);
    } else {
        startNewGame(false);
    }
    setInterval(poll, POLL_INTERVAL_MS);
});
