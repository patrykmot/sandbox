"""Tree-count arithmetic for the Isolation Forest.

`_calculate_n_estimators` decides how many trees a run trains with, and it
once decided "one" for every realistic input - a forest of one tree scores by
the luck of its random splits, so the supervisor's outlier test passed or
failed at random. These are the cases that pin it down.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from src.implementations.iso_forest_detector import (
    MIN_TREES,
    IsolationForestAnomalyDetector,
)
from src.interfaces.encoder import FeatureVector

calculate = IsolationForestAnomalyDetector._calculate_n_estimators


# --- The floor ---------------------------------------------------------------------------


def test_every_tree_seeing_every_row_still_builds_a_forest() -> None:
    """S == N satisfies coverage with one tree, and the formula would take
    ln(0). Neither is a reason to ship a one-tree forest."""
    assert calculate(dataset_size=64, max_samples=256) == MIN_TREES
    assert calculate(dataset_size=256, max_samples=256) == MIN_TREES


def test_just_past_that_boundary_is_not_a_single_tree_either() -> None:
    """N = S + 1: coverage is reached almost immediately, so the formula's own
    answer here is 1. The floor is what stops it."""
    assert math.ceil(
        math.log(1 - 0.99) / math.log(1 - 256 / 257)
    ) == 1, "premise changed: the raw formula no longer returns 1 here"

    assert calculate(dataset_size=257, max_samples=256) == MIN_TREES


def test_a_large_dataset_gets_what_coverage_asks_for() -> None:
    """The floor is a floor, not a constant - past ~5700 samples coverage
    takes over and the count climbs."""
    assert calculate(dataset_size=20_000, max_samples=256) == 358
    assert calculate(dataset_size=100_000, max_samples=256) == 1797


def test_more_coverage_never_asks_for_fewer_trees() -> None:
    counts = [
        calculate(dataset_size=50_000, max_samples=256, coverage_percent=c)
        for c in (50, 90, 99, 99.9, 99.99)
    ]
    assert counts == sorted(counts), counts


# --- Arguments ---------------------------------------------------------------------------


def test_full_coverage_is_refused_rather_than_clamped() -> None:
    """Random subsampling never guarantees every row is seen: 100% is a limit
    needing infinitely many trees, and ln(0) says so. Refuse it instead of
    inventing a cap."""
    with pytest.raises(ValueError, match="infinitely"):
        calculate(dataset_size=1000, coverage_percent=100)


def test_coverage_outside_the_range_is_refused() -> None:
    for bad in (0, -5, 100.1, 1000):
        with pytest.raises(ValueError):
            calculate(dataset_size=1000, coverage_percent=bad)


def test_an_empty_or_impossible_dataset_is_refused() -> None:
    with pytest.raises(ValueError):
        calculate(dataset_size=0)
    with pytest.raises(ValueError):
        calculate(dataset_size=1000, max_samples=0)


def test_a_fraction_sized_coverage_is_legal_but_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """0.99 is a valid 0.99% request and no longer collapses to one tree - the
    floor absorbs it. That is exactly why it has to be said out loud, or the
    original mistake would repeat in silence."""
    with caplog.at_level("WARNING"):
        assert calculate(dataset_size=20_000, coverage_percent=0.99) == MIN_TREES

    assert "less than 1%" in caplog.text
    assert "99" in caplog.text  # tells the caller what they probably meant


# --- Wiring ------------------------------------------------------------------------------


def test_the_constructor_default_reaches_the_forest() -> None:
    """The bug was a unit mismatch between the constructor and the method, so
    the default has to be followed all the way to the fitted model."""
    rng = random.Random(42)
    training_data = [
        [
            FeatureVector(
                t=i,
                vector=np.array([100 + rng.uniform(-2, 2), 500 + rng.uniform(-5, 5)]),
            )
        ]
        for i in range(64)
    ]

    detector = IsolationForestAnomalyDetector()
    detector.fit(training_data)

    assert detector._model is not None
    assert detector._model.n_estimators == MIN_TREES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
