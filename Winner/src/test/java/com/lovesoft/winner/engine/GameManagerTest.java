package com.lovesoft.winner.engine;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GameManagerTest {

    private final GameManager manager = new GameManager();

    @Test
    void createsStartedGamesWithDefaultOrCustomNames() {
        Game defaults = manager.createHumanGame(null, "  ");
        assertEquals(GameStatus.IN_PROGRESS, defaults.getStatus());
        assertEquals("Player 1", defaults.snapshot().whitePlayer());
        assertEquals("Player 2", defaults.snapshot().blackPlayer());

        Game named = manager.createHumanGame("Alice", "Bob");
        assertEquals("Alice", named.snapshot().whitePlayer());
        assertEquals("Bob", named.snapshot().blackPlayer());

        assertNotEquals(defaults.getId(), named.getId());
        assertEquals(2, manager.size());
    }

    @Test
    void looksUpGamesById() {
        Game game = manager.createHumanGame(null, null);
        assertTrue(manager.getGame(game.getId()).isPresent());
        assertTrue(manager.getGame(game.getId().toString()).isPresent());
        assertFalse(manager.getGame(UUID.randomUUID()).isPresent());
        assertThrows(IllegalArgumentException.class, () -> manager.getGame("not-a-uuid"));
    }

    @Test
    void parallelGamesStayIndependent() throws Exception {
        int games = 16;
        List<Game> created = new ArrayList<>();
        for (int i = 0; i < games; i++) {
            created.add(manager.createHumanGame(null, null));
        }

        ExecutorService pool = Executors.newFixedThreadPool(8);
        try {
            List<Future<GameSnapshot>> results = new ArrayList<>();
            for (Game game : created) {
                results.add(pool.submit(() -> {
                    game.submitMove(MoveRequest.ofSan("e4"));
                    game.submitMove(MoveRequest.ofSan("e5"));
                    return game.submitMove(MoveRequest.ofSan("Nf3"));
                }));
            }
            for (Future<GameSnapshot> result : results) {
                GameSnapshot s = result.get(10, TimeUnit.SECONDS);
                assertEquals(List.of("e4", "e5", "Nf3"), s.moveHistory());
                assertEquals("BLACK", s.turn());
            }
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void concurrentSubmissionsOfTheSameMoveSucceedOnlyOnce() throws Exception {
        Game game = manager.createHumanGame(null, null);
        int threads = 8;
        CountDownLatch go = new CountDownLatch(1);
        AtomicInteger successes = new AtomicInteger();
        AtomicInteger rejections = new AtomicInteger();

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                futures.add(pool.submit(() -> {
                    go.await();
                    try {
                        game.submitMove(MoveRequest.of("e2", "e4"));
                        successes.incrementAndGet();
                    } catch (IllegalMoveException e) {
                        rejections.incrementAndGet(); // second e2-e4 is illegal: black is to move
                    }
                    return null;
                }));
            }
            go.countDown();
            for (Future<?> f : futures) {
                f.get(10, TimeUnit.SECONDS);
            }
        } finally {
            pool.shutdownNow();
        }

        assertEquals(1, successes.get());
        assertEquals(threads - 1, rejections.get());
        assertEquals(List.of("e4"), game.snapshot().moveHistory());
    }
}
