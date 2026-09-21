"""Does the PyTorch autoencoder detector actually detect?

Skipped automatically where torch isn't installed, so the rest of the suite
still runs; install requirements.txt to exercise them.

The scenario throughout: a "normal" scene of objects milling about with small
jitter, and one frame that is nothing like it. Training sees only the normal
frames, so a detector that works flags the odd one and leaves the rest alone.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch not installed")

from src.implementations.autoencoder_detector import (  # noqa: E402
    AutoencoderNet,
    PyTorchAutoencoderDetector,
)
from src.interfaces.encoder import FeatureVector  # noqa: E402

FEATURES = 6


def make_normal_frames(count: int, seed: int = 42) -> list[list[FeatureVector]]:
    """Frames drawn from one tight distribution - the baseline to learn."""
    rng = random.Random(seed)
    frames = []
    for i in range(count):
        t = 1_700_000_000_000 + i * 33
        vector = np.array(
            [
                2.0 + rng.uniform(-0.2, 0.2),
                150.0 + rng.uniform(-5, 5),
                300.0 + rng.uniform(-8, 8),
                12.0 + rng.uniform(-1, 1),
                480.0 + rng.uniform(-10, 10),
                5_000.0 + rng.uniform(-100, 100),
            ],
            dtype=np.float64,
        )
        frames.append([FeatureVector(t=t, vector=vector)])
    return frames


def make_outlier_frame(t: int = 1_700_000_999_999) -> list[FeatureVector]:
    """Nothing like the baseline: far more objects, far faster, far bigger."""
    return [
        FeatureVector(
            t=t,
            vector=np.array([40.0, 9_000.0, 12_000.0, 3_000.0, 25_000.0, 900_000.0]),
        )
    ]


def train_detector(**kwargs) -> PyTorchAutoencoderDetector:
    detector = PyTorchAutoencoderDetector(epochs=kwargs.pop("epochs", 60), **kwargs)
    detector.fit(make_normal_frames(200))
    return detector


# --- The network ------------------------------------------------------------------------


def test_autoencoder_net_round_trips_its_input_shape() -> None:
    """The bottleneck must not change the shape that comes back out, or the
    reconstruction error would be meaningless."""
    import torch

    net = AutoencoderNet(input_dim=FEATURES, hidden_dim=8, latent_dim=3)
    output = net(torch.zeros(4, FEATURES))

    assert output.shape == (4, FEATURES)


# --- Training ---------------------------------------------------------------------------


def test_fit_trains_and_learns_a_threshold() -> None:
    detector = train_detector()

    assert detector.is_trained()
    assert detector.maximum_normal_error is not None
    assert detector.maximum_normal_error > 0


def test_fit_rejects_empty_training_data() -> None:
    with pytest.raises(ValueError):
        PyTorchAutoencoderDetector(epochs=1).fit([])


def test_predict_before_fit_is_an_error_not_a_guess() -> None:
    with pytest.raises(RuntimeError):
        PyTorchAutoencoderDetector().predict(make_outlier_frame())


def test_percentile_threshold_must_be_a_fraction() -> None:
    """99 instead of 0.99 would put the line near the *minimum* error and
    flag every frame. Refuse it rather than detect everything."""
    with pytest.raises(ValueError):
        PyTorchAutoencoderDetector(percentile_threshold=99)
    with pytest.raises(ValueError):
        PyTorchAutoencoderDetector(percentile_threshold=0.0)


def test_progress_callback_climbs_to_100() -> None:
    seen: list[int] = []
    PyTorchAutoencoderDetector(epochs=5).fit(
        make_normal_frames(60), progress_callback=seen.append
    )

    assert seen, "training reported nothing at all"
    assert seen == sorted(seen), f"progress went backwards: {seen}"
    assert seen[-1] == 100
    assert all(0 <= percent <= 100 for percent in seen)


def test_a_failing_callback_does_not_abort_training() -> None:
    """The callback is a UI's hook. A UI that throws is the UI's problem."""

    def exploding(percent: int) -> None:
        raise RuntimeError("dashboard exploded")

    detector = PyTorchAutoencoderDetector(epochs=3)
    detector.fit(make_normal_frames(60), progress_callback=exploding)

    assert detector.is_trained()


# --- Detection --------------------------------------------------------------------------


def test_normal_frames_are_not_flagged() -> None:
    detector = train_detector()

    # Unseen frames from the same distribution: the whole point is that the
    # model generalises to these, not that it memorised the training set.
    flagged = [
        detector.predict(frame)[0] for frame in make_normal_frames(30, seed=99)
    ]

    # The threshold is the 99th percentile of training error, so an occasional
    # normal frame landing above it is expected - a majority doing so is not.
    assert sum(flagged) <= 3, f"{sum(flagged)}/30 normal frames flagged"


def test_an_outlier_frame_is_flagged() -> None:
    detector = train_detector()

    is_anomaly, score = detector.predict(make_outlier_frame())

    assert is_anomaly
    assert score > detector.maximum_normal_error


def test_the_returned_score_is_the_error_the_decision_was_made_on() -> None:
    """is_anomaly and the score must agree - the panel shows the score as the
    reason for the alarm."""
    detector = train_detector()

    for frame in (make_outlier_frame(), make_normal_frames(1, seed=7)[0]):
        is_anomaly, score = detector.predict(frame)
        assert is_anomaly == (score > detector.maximum_normal_error)


def test_an_empty_frame_is_not_an_anomaly() -> None:
    """Nothing detected this frame - nothing to flag."""
    detector = train_detector()

    assert detector.predict([]) == (False, 0.0)


def test_standardization_keeps_a_huge_column_from_drowning_the_others() -> None:
    """With a NO_PAIR_DISTANCE-sized column present, an anomaly in a *small*
    column must still register. Raw MSE would bury it under the big one."""
    rng = random.Random(3)
    frames = []
    for i in range(200):
        frames.append(
            [
                FeatureVector(
                    t=i,
                    vector=np.array(
                        [2.0 + rng.uniform(-0.1, 0.1), 7_777_777.0 + rng.uniform(-5, 5)]
                    ),
                )
            ]
        )

    detector = PyTorchAutoencoderDetector(hidden_dim=8, latent_dim=2, epochs=80)
    detector.fit(frames)

    # The big column is untouched; only the small one is out of character.
    odd = [FeatureVector(t=1, vector=np.array([50.0, 7_777_777.0]))]
    assert detector.predict(odd)[0]


# --- Persistence ------------------------------------------------------------------------


def test_save_and_load_round_trip_gives_identical_predictions(tmp_path) -> None:
    detector = train_detector()
    path = str(tmp_path / "models" / "autoencoder.pt")   # a directory that does not exist yet

    detector.save(path)
    restored = PyTorchAutoencoderDetector()
    restored.load(path)

    assert restored.is_trained()
    assert restored.maximum_normal_error == pytest.approx(detector.maximum_normal_error)

    for frame in (make_outlier_frame(), *make_normal_frames(5, seed=11)):
        frame = frame if isinstance(frame, list) else [frame]
        assert restored.predict(frame)[1] == pytest.approx(detector.predict(frame)[1])
        assert restored.predict(frame)[0] == detector.predict(frame)[0]


def test_saving_an_untrained_model_is_refused(tmp_path) -> None:
    """A file that looks like a model but predicts nothing is worse than no file."""
    with pytest.raises(RuntimeError):
        PyTorchAutoencoderDetector().save(str(tmp_path / "empty.pt"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
