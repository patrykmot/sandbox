"""Tests for BiologicalFeatureEncoder.

The encoder is FrameMergingFeatureEncoder's vector with object slots glued on
the end, so the tests check the two halves separately: the global part must
be exactly what the parent produces, and the slot part is checked by hand
with the default trait table from src.config.

Run the walkthrough:

    pytest tests/test_biological_feature_encoder.py -s
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.config import DEFAULT_OBJECT_TRAITS, Config, FeatureEncoderKind
from src.implementations.biological_feature_encoder import BiologicalFeatureEncoder
from src.implementations.frame_merging_feature_encoder import FrameMergingFeatureEncoder
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector
from src.interfaces.encoder import StateVector

T1 = 1_700_000_000_000
T2 = 1_700_000_000_033

BASE_DIM = FrameMergingFeatureEncoder.FEATURE_DIM  # 17
SLOT_DIM = 16  # present, x, y, vx, vy, size + 10 traits

# COCO ids: 0 = person, 2 = car, 16 = dog, 56 = chair.
PERSON = StateVector(t=T1, x=10.0, y=20.0, vx=1.0, vy=2.0, size=300.0, object_type=0)
CAR = StateVector(t=T1, x=100.0, y=50.0, vx=-30.0, vy=0.0, size=5_000.0, object_type=2)
DOG = StateVector(t=T1, x=40.0, y=60.0, vx=5.0, vy=-5.0, size=150.0, object_type=16)


def make_encoder(max_objects: int = 5) -> BiologicalFeatureEncoder:
    return BiologicalFeatureEncoder(traits=DEFAULT_OBJECT_TRAITS, max_objects=max_objects)


def slots_of(vector: np.ndarray, encoder: BiologicalFeatureEncoder) -> np.ndarray:
    """The appended part of the vector, one row per slot."""
    return vector[BASE_DIM:].reshape(encoder.max_objects, encoder.slot_dim)


def traits(*names: str) -> list[float]:
    """Expected trait columns: 1.0 for each named trait, in config order."""
    return [1.0 if name in names else 0.0 for name in DEFAULT_OBJECT_TRAITS]


def test_dimensions_with_default_config() -> None:
    encoder = make_encoder()

    assert encoder.slot_dim == SLOT_DIM
    assert encoder.FEATURE_DIM == BASE_DIM + 5 * SLOT_DIM == 97
    assert len(encoder.FEATURE_NAMES) == encoder.FEATURE_DIM
    assert len(set(encoder.FEATURE_NAMES)) == encoder.FEATURE_DIM  # no duplicates
    assert encoder.FEATURE_NAMES[:BASE_DIM] == FrameMergingFeatureEncoder.FEATURE_NAMES
    assert encoder.FEATURE_NAMES[BASE_DIM : BASE_DIM + 7] == (
        "obj0_present", "obj0_x", "obj0_y", "obj0_vx", "obj0_vy", "obj0_size", "obj0_trait_biological",
    )
    assert encoder.FEATURE_NAMES[-1] == "obj4_trait_handheld"

    # The parent's class-level names are untouched by the instance override.
    assert FrameMergingFeatureEncoder.FEATURE_DIM == 17


def test_global_part_is_exactly_the_parent_vector() -> None:
    parent = FrameMergingFeatureEncoder().encode([PERSON, CAR, DOG])[0].vector
    enriched = make_encoder().encode([PERSON, CAR, DOG])[0].vector

    assert enriched.shape == (97,)
    np.testing.assert_array_equal(enriched[:BASE_DIM], parent)


def test_slots_are_sorted_by_size_descending_and_padded_with_zeros() -> None:
    encoder = make_encoder()
    # Input deliberately NOT in size order.
    features = encoder.encode([DOG, PERSON, CAR])
    assert len(features) == 1
    assert features[0].t == T1

    slots = slots_of(features[0].vector, encoder)

    # car (5000) > person (300) > dog (150)
    assert slots[0].tolist() == [1.0, 100.0, 50.0, -30.0, 0.0, 5_000.0, *traits("MOTORIZED", "RIDEABLE", "HEAVY", "WHEELED")]
    assert slots[1].tolist() == [1.0, 10.0, 20.0, 1.0, 2.0, 300.0, *traits("BIOLOGICAL")]
    assert slots[2].tolist() == [1.0, 40.0, 60.0, 5.0, -5.0, 150.0, *traits("BIOLOGICAL")]

    # Two unused slots: every one of their 16 columns is exactly 0.0.
    assert np.all(slots[3:] == 0.0)


def test_more_objects_than_slots_keeps_only_the_largest() -> None:
    encoder = make_encoder(max_objects=2)
    slots = slots_of(encoder.encode([DOG, PERSON, CAR])[0].vector, encoder)

    assert slots.shape == (2, SLOT_DIM)
    assert slots[:, 5].tolist() == [5_000.0, 300.0]  # dog (150) dropped
    # ...but the global part still counts all three.
    assert encoder.encode([DOG, PERSON, CAR])[0].vector[0] == 3.0


def test_multi_hot_traits_for_selected_classes() -> None:
    encoder = make_encoder()

    np.testing.assert_array_equal(encoder.traits_of(0), traits("BIOLOGICAL"))  # person
    np.testing.assert_array_equal(encoder.traits_of(1), traits("RIDEABLE", "WHEELED"))  # bicycle
    np.testing.assert_array_equal(  # airplane
        encoder.traits_of(4), traits("MOTORIZED", "RIDEABLE", "HEAVY", "AERIAL", "WHEELED")
    )
    np.testing.assert_array_equal(encoder.traits_of(14), traits("BIOLOGICAL", "AERIAL"))  # bird
    np.testing.assert_array_equal(  # horse
        encoder.traits_of(17), traits("BIOLOGICAL", "RIDEABLE")
    )
    np.testing.assert_array_equal(encoder.traits_of(56), traits("FURNITURE"))  # chair
    np.testing.assert_array_equal(encoder.traits_of(67), traits("PORTABLE", "HANDHELD"))  # cell phone
    # An id no trait mentions is still a valid object - just with no traits.
    np.testing.assert_array_equal(encoder.traits_of(999), traits())


def test_equal_sizes_give_the_same_slots_whatever_the_input_order() -> None:
    a = StateVector(t=T1, x=1.0, y=0.0, vx=0.0, vy=0.0, size=100.0, object_type=0)
    b = StateVector(t=T1, x=2.0, y=0.0, vx=0.0, vy=0.0, size=100.0, object_type=2)
    encoder = make_encoder()

    np.testing.assert_array_equal(encoder.encode([a, b])[0].vector, encoder.encode([b, a])[0].vector)


def test_groups_by_timestamp_like_the_parent() -> None:
    later = StateVector(t=T2, x=5.0, y=5.0, vx=0.0, vy=0.0, size=10.0, object_type=56)
    encoder = make_encoder()

    features = encoder.encode([later, PERSON])

    assert [fv.t for fv in features] == [T1, T2]
    assert slots_of(features[1].vector, encoder)[0].tolist() == [1.0, 5.0, 5.0, 0.0, 0.0, 10.0, *traits("FURNITURE")]
    assert encoder.encode([]) == []


def test_trait_table_and_slot_count_come_from_configuration() -> None:
    encoder = BiologicalFeatureEncoder(traits={"ALIVE": [0, 16], "CAR": [2]}, max_objects=3)

    assert encoder.slot_dim == 8
    assert encoder.FEATURE_DIM == BASE_DIM + 3 * 8
    slots = slots_of(encoder.encode([CAR, DOG])[0].vector, encoder)
    assert slots[0, 6:].tolist() == [0.0, 1.0]
    assert slots[1, 6:].tolist() == [1.0, 0.0]


def test_invalid_slot_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        BiologicalFeatureEncoder(traits=DEFAULT_OBJECT_TRAITS, max_objects=0)


def test_config_switch_builds_the_right_encoder() -> None:
    # Imported here, not at module level: main pulls in the whole runtime
    # (YOLO, torch, FastAPI), which the encoder tests above don't need.
    from main import build_feature_encoder

    frame_merging = Config(feature_encoder=FeatureEncoderKind.FRAME_MERGING)
    assert type(build_feature_encoder(frame_merging)) is FrameMergingFeatureEncoder

    encoder = build_feature_encoder(
        Config(feature_encoder=FeatureEncoderKind.BIOLOGICAL, biological_max_objects=3)
    )
    assert isinstance(encoder, BiologicalFeatureEncoder)
    assert encoder.max_objects == 3
    assert encoder.trait_names == tuple(DEFAULT_OBJECT_TRAITS)


def test_feature_encoder_setting_accepts_env_strings_and_rejects_unknown_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATURE_ENCODER", "biological")
    assert Config().feature_encoder is FeatureEncoderKind.BIOLOGICAL

    # A typo fails when the config loads, not halfway through a run.
    monkeypatch.setenv("FEATURE_ENCODER", "biologcal")
    with pytest.raises(ValidationError):
        Config()


def test_default_config_keeps_the_existing_encoder() -> None:
    assert Config().feature_encoder is FeatureEncoderKind.FRAME_MERGING
    assert Config().biological_max_objects == 5
    assert Config().biological_object_traits == DEFAULT_OBJECT_TRAITS


def test_all_features_finite_so_isolation_forest_accepts_them() -> None:
    encoder = make_encoder()
    frames = []
    for i in range(30):
        t = T1 + i * 33
        objects = [StateVector(t=t, x=float(i), y=0.0, vx=1.0, vy=0.0, size=100.0 + i, object_type=0)]
        if i % 2:
            objects.append(StateVector(t=t, x=float(i) + 5.0, y=0.0, vx=2.0, vy=1.0, size=150.0, object_type=2))
        frames.append(encoder.encode(objects))

    assert all(np.all(np.isfinite(fv.vector)) for frame in frames for fv in frame)

    detector = IsolationForestAnomalyDetector()
    detector.fit(frames)
    assert detector._model.n_features_in_ == encoder.FEATURE_DIM


def test_walkthrough_print() -> None:
    """Prints the slot part with labels for `pytest -s`."""
    encoder = make_encoder()
    vector = encoder.encode([PERSON, CAR, DOG])[0].vector

    print("\n--- BiologicalFeatureEncoder: object slots (global part as in FrameMerging) ---")
    for name, value in zip(encoder.FEATURE_NAMES[BASE_DIM:], vector[BASE_DIM:]):
        if not name.startswith(("obj3_", "obj4_")):
            print(f"  {name:<28} = {value:>10.2f}")

    assert vector.shape == (encoder.FEATURE_DIM,)
