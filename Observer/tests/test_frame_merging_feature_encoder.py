"""Tests for FrameMergingFeatureEncoder.

The main case uses two objects with deliberately clean numbers (a 3-4-5
triangle) so every expected value can be checked by hand rather than being
copied from whatever the code happened to produce.

Run the walkthrough:

    pytest tests/test_frame_merging_feature_encoder.py -s
"""

from __future__ import annotations

import numpy as np

from src.implementations.frame_merging_feature_encoder import FrameMergingFeatureEncoder
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector
from src.interfaces.encoder import StateVector

T1 = 1_700_000_000_000
T2 = 1_700_000_000_033

# Two objects, chosen so the pairwise maths is exact:
#   positions (0,0) and (3,4)      -> distance 5.0
#   velocities (0,0) and (3,4)     -> relative speed 5.0
#   speeds                         -> 0.0 and 5.0
#   sizes                          -> 100.0 and 200.0
OBJ_A = StateVector(t=T1, x=0.0, y=0.0, vx=0.0, vy=0.0, size=100.0, object_type=0)
OBJ_B = StateVector(t=T1, x=3.0, y=4.0, vx=3.0, vy=4.0, size=200.0, object_type=2)


def index_of(name: str) -> int:
    return FrameMergingFeatureEncoder.FEATURE_NAMES.index(name)


def as_dict(vector: np.ndarray) -> dict[str, float]:
    return dict(zip(FrameMergingFeatureEncoder.FEATURE_NAMES, vector))


def test_feature_names_match_vector_width() -> None:
    assert FrameMergingFeatureEncoder.FEATURE_DIM == 17
    assert len(FrameMergingFeatureEncoder.FEATURE_NAMES) == 17
    assert len(set(FrameMergingFeatureEncoder.FEATURE_NAMES)) == 17  # no duplicates


def test_two_objects_produce_hand_checkable_values() -> None:
    features = FrameMergingFeatureEncoder().encode([OBJ_A, OBJ_B])

    assert len(features) == 1
    assert features[0].t == T1
    assert features[0].vector.shape == (17,)

    f = as_dict(features[0].vector)

    assert f["num_objects"] == 2.0

    # Exactly one pair, so min == p10 == p50 == p90 == that pair's distance.
    assert f["dist_min"] == 5.0
    assert f["dist_p10"] == 5.0
    assert f["dist_p50"] == 5.0
    assert f["dist_p90"] == 5.0

    # Likewise for the single relative-speed value.
    assert f["rel_speed_p10"] == 5.0
    assert f["rel_speed_p50"] == 5.0
    assert f["rel_speed_p90"] == 5.0
    assert f["rel_speed_max"] == 5.0

    # Per-object speeds are [0.0, 5.0]; numpy interpolates linearly.
    assert f["speed_p10"] == 0.5
    assert f["speed_p50"] == 2.5
    assert f["speed_p90"] == 4.5
    assert f["speed_max"] == 5.0

    # Sizes are [100.0, 200.0].
    assert f["size_p10"] == 110.0
    assert f["size_p50"] == 150.0
    assert f["size_p90"] == 190.0
    assert f["size_max"] == 200.0


def test_groups_by_timestamp_and_sorts_ascending() -> None:
    """Two frames' worth of objects, deliberately interleaved and out of order."""
    later = StateVector(t=T2, x=10.0, y=0.0, vx=1.0, vy=0.0, size=50.0, object_type=0)
    other_later = StateVector(t=T2, x=20.0, y=0.0, vx=1.0, vy=0.0, size=60.0, object_type=0)

    # T2 entries come first in the input; output must still be time-ordered.
    features = FrameMergingFeatureEncoder().encode([later, OBJ_A, other_later, OBJ_B])

    assert [fv.t for fv in features] == [T1, T2]

    assert as_dict(features[0].vector)["num_objects"] == 2.0
    assert as_dict(features[1].vector)["num_objects"] == 2.0
    # The two T2 objects are 10 apart on the x axis.
    assert as_dict(features[1].vector)["dist_min"] == 10.0


def test_single_object_uses_sentinel_distance_not_zero() -> None:
    """One object has no pairs - but its own speed/size are still real values."""
    solo = StateVector(t=T1, x=7.0, y=0.0, vx=3.0, vy=4.0, size=800.0, object_type=0)

    features = FrameMergingFeatureEncoder().encode([solo])
    assert len(features) == 1

    f = as_dict(features[0].vector)

    assert f["num_objects"] == 1.0

    # No neighbour means "infinitely far", NOT zero - 0.0 has to stay
    # reserved for objects that genuinely almost touch.
    sentinel = FrameMergingFeatureEncoder.NO_PAIR_DISTANCE
    assert sentinel == 7_777_777.0
    for name in FrameMergingFeatureEncoder.FEATURE_NAMES:
        if name.startswith("dist_"):
            assert f[name] == sentinel, name

    # Relative speed keeps 0.0: with no second object there is no relative motion.
    for name in FrameMergingFeatureEncoder.FEATURE_NAMES:
        if name.startswith("rel_speed_"):
            assert f[name] == 0.0, name

    # Single-object stats are still genuinely measured: ||(3,4)|| == 5.
    assert f["speed_p50"] == 5.0
    assert f["speed_max"] == 5.0
    assert f["size_p50"] == 800.0
    assert f["size_max"] == 800.0


def test_touching_objects_stay_near_zero_and_far_from_the_sentinel() -> None:
    """The whole point of the sentinel: near-zero distance must remain distinguishable."""
    a = StateVector(t=T1, x=100.0, y=100.0, vx=0.0, vy=0.0, size=500.0, object_type=0)
    b = StateVector(t=T1, x=100.5, y=100.0, vx=0.0, vy=0.0, size=500.0, object_type=0)

    f = as_dict(FrameMergingFeatureEncoder().encode([a, b])[0].vector)

    assert f["num_objects"] == 2.0
    assert f["dist_min"] == 0.5
    assert f["dist_min"] < FrameMergingFeatureEncoder.NO_PAIR_DISTANCE


def test_empty_input_produces_no_features() -> None:
    assert FrameMergingFeatureEncoder().encode([]) == []


def test_all_features_finite_so_isolation_forest_accepts_them() -> None:
    """Guards the reason pairwise gaps are 0.0 and not NaN: sklearn rejects non-finite input."""
    encoder = FrameMergingFeatureEncoder()

    frames = []
    for i in range(30):
        t = T1 + i * 33
        # Alternate between one-object and two-object frames so the K=1
        # zero-filled path lands in the training data too.
        if i % 2 == 0:
            frames.append(encoder.encode([StateVector(t=t, x=float(i), y=0.0, vx=1.0, vy=0.0, size=100.0, object_type=0)]))
        else:
            frames.append(
                encoder.encode(
                    [
                        StateVector(t=t, x=float(i), y=0.0, vx=1.0, vy=0.0, size=100.0, object_type=0),
                        StateVector(t=t, x=float(i) + 5.0, y=0.0, vx=2.0, vy=1.0, size=150.0, object_type=2),
                    ]
                )
            )

    assert all(np.all(np.isfinite(fv.vector)) for frame in frames for fv in frame)

    detector = IsolationForestAnomalyDetector()
    detector.fit(frames)

    assert detector.is_trained() is True
    assert detector._model.n_features_in_ == FrameMergingFeatureEncoder.FEATURE_DIM

    is_anomaly, score = detector.predict(frames[-1])
    assert isinstance(is_anomaly, bool)
    assert isinstance(score, float)


def test_walkthrough_print() -> None:
    """Not an assertion-heavy test - prints the vector with labels for `pytest -s`."""
    features = FrameMergingFeatureEncoder().encode([OBJ_A, OBJ_B])
    f = as_dict(features[0].vector)

    print("\n--- FrameMergingFeatureEncoder: 2 objects merged into 1 FeatureVector ---")
    print(f"  t = {features[0].t}")
    for name, value in f.items():
        print(f"  {name:<16} = {value:>10.2f}")

    assert len(f) == 17
