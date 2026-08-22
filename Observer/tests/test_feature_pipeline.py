"""A walkthrough of the encoding -> training flow, written to be stepped through.

    list[StateVector] -> IFeatureEncoder -> list[FeatureVector] -> IAnomalyDetector.fit()

Everything here is spelled out with literal values rather than helpers or
loops, so you can set a breakpoint anywhere and see real numbers.

Run it on its own, showing the printed walkthrough:

    pytest tests/test_feature_pipeline.py -s

This file is a plain pytest test (a function, not a unittest.TestCase), so
it must be run with pytest. In PyCharm set
    Settings -> Tools -> Python Integrated Tools -> Testing -> Default test runner: pytest
or just right-click this file and use Run/Debug, which executes the
__main__ block at the bottom - no test runner involved at all.

Good places to put a breakpoint:
  * DummyFeatureEncoder.encode()               - see each FeatureVector built
  * IsolationForestAnomalyDetector.fit()       - see the training matrix
  * IsolationForestAnomalyDetector._stack()    - see list -> numpy conversion
"""

from __future__ import annotations

import numpy as np

from src.implementations.dummy_feature_encoder import DummyFeatureEncoder
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector
from src.interfaces.encoder import StateVector

# One shared timestamp: these three objects were all seen in the SAME frame.
T = 1_700_000_000_000

# Three different objects, using COCO class ids (0=person, 2=car, 16=dog).
#                        t     x      y      vx      vy    size     type
PERSON = StateVector(t=T, x=100.0, y=200.0, vx=15.0, vy=-5.0, size=8_000.0, object_type=0)
CAR = StateVector(t=T, x=640.0, y=360.0, vx=-120.0, vy=0.0, size=45_000.0, object_type=2)
DOG = StateVector(t=T, x=310.0, y=410.0, vx=30.0, vy=12.0, size=2_500.0, object_type=16)


def test_state_vectors_become_feature_vectors_and_train_the_detector() -> None:
    # ---------------------------------------------------------------- STEP 1
    # What the video encoder would have produced for a single frame:
    # three tracked objects, all sharing the same timestamp.
    states: list[StateVector] = [PERSON, CAR, DOG]

    assert len(states) == 3
    assert all(state.t == T for state in states)

    # ---------------------------------------------------------------- STEP 2
    # IFeatureEncoder turns kinematic states into plain float vectors.
    # DummyFeatureEncoder is the Phase I placeholder: it maps 1 StateVector
    # to exactly 1 FeatureVector, laid out as
    #     [x, y, vx, vy, size, object_type]
    # (A real encoder may return a different number of FeatureVectors than
    # it received - the interface explicitly allows that.)
    encoder = DummyFeatureEncoder()
    features = encoder.encode(states)

    assert len(features) == 3

    # The timestamp is carried straight through from the StateVector, which
    # is how a FeatureVector stays traceable back to when it happened.
    assert [fv.t for fv in features] == [T, T, T]

    # Each vector is the 6 numbers from its StateVector, in order.
    assert np.array_equal(features[0].vector, [100.0, 200.0, 15.0, -5.0, 8_000.0, 0.0])
    assert np.array_equal(features[1].vector, [640.0, 360.0, -120.0, 0.0, 45_000.0, 2.0])
    assert np.array_equal(features[2].vector, [310.0, 410.0, 30.0, 12.0, 2_500.0, 16.0])

    # Every FeatureVector.vector must be the same length - that length is the
    # number of columns the Isolation Forest will be trained on.
    assert all(fv.vector.shape == (6,) for fv in features)

    # ---------------------------------------------------------------- STEP 3
    # fit() takes a list of FRAMES, where each frame is a list of
    # FeatureVectors. We only have one frame here, hence the extra brackets:
    #
    #     [ [ person, car, dog ] ]
    #       ^ frame 1  ^ its 3 objects
    #
    # Internally fit() flattens every frame into one big (n_samples,
    # n_features) matrix - so ONE Isolation Forest is trained on ALL objects
    # from ALL frames, which is what "one forest for the entire data" means.
    training_data: list[list] = [features]

    detector = IsolationForestAnomalyDetector()
    assert detector.is_trained() is False

    detector.fit(training_data)

    assert detector.is_trained() is True

    # This is the exact matrix fit() built and handed to sklearn:
    # 3 rows (one per object) x 6 columns (one per feature).
    training_matrix = np.stack([fv.vector for fv in features])
    assert training_matrix.shape == (3, 6)
    assert detector._model.n_features_in_ == 6

    # ---------------------------------------------------------------- STEP 4
    # Sanity check that the trained model is usable: predict() takes the
    # list of FeatureVectors for one frame and collapses them into a single
    # (is_anomaly, score) answer for that frame.
    #
    # HEADS UP - this frame comes back as an ANOMALY even though it is the
    # exact data we just trained on. That is expected here, not a bug:
    # contamination=0.05 tells sklearn "assume ~5% of the training data is
    # abnormal", and it places the decision threshold accordingly. With only
    # 3 samples that threshold has to fall somewhere, so at least one of the
    # three gets labelled abnormal. Three points simply aren't enough to
    # describe "normal" - which is exactly why the real system collects
    # COLLECTION_TARGET_VALUE (1000) frames before calling fit().
    is_anomaly, score = detector.predict(features)

    assert isinstance(is_anomaly, bool)
    assert isinstance(score, float)

    # --------------------------------------------------------- walkthrough
    # Visible with `pytest -s`; pytest hides this on a passing run otherwise.
    # suppress=True keeps the numbers readable instead of scientific notation.
    with np.printoptions(suppress=True, precision=1, linewidth=100):
        print("\n--- StateVector -> FeatureVector ---")
        for state, feature in zip(states, features):
            print(f"  type={state.object_type:>2}  t={state.t}  ->  {feature.vector}")

        print("\n--- Training matrix handed to IsolationForest.fit() ---")
        print(f"  shape = {training_matrix.shape}  (rows=objects, cols=features)")
        print("  columns = [x, y, vx, vy, size, object_type]")
        print(training_matrix)

    print("\n--- predict() on that same frame ---")
    print(f"  is_anomaly={is_anomaly}  score={score:.4f}")
    print("  (anomaly on training data is expected with only 3 samples - see STEP 4)")


if __name__ == "__main__":
    # Lets you Run/Debug this file directly as a plain script - breakpoints
    # work, and no pytest/unittest runner configuration is needed.
    test_state_vectors_become_feature_vectors_and_train_the_detector()
    print("\nOK - all assertions passed.")
