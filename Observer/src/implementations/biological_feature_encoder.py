"""IFeatureEncoder that fuses global frame statistics with per-object slots.

FrameMergingFeatureEncoder squeezes a whole frame into percentiles: good at
describing the crowd, blind to *who* is in it - a dog and a bus of the same
size moving at the same speed are indistinguishable there. This encoder keeps
that global vector untouched and appends a fixed number of object "slots",
one per largest object, each carrying that object's raw physics plus a
multi-hot description of what kind of thing it is:

    [ global stats (17) | slot 0 | slot 1 | ... | slot MAX_OBJECTS-1 ]

    slot = [is_present, x, y, vx, vy, size, trait_1, ..., trait_K]

Why traits instead of the raw COCO class id: the id is a label, not a
quantity - class 17 is not "between" 16 and 18, so feeding it as a number
invents an order the detector would happily learn. Traits ("biological",
"motorized", "heavy", ...) are binary and shareable across classes, so a
horse and a cow land close together while a horse and a truck do not. The
trait table lives in configuration (Config.biological_object_traits), so
which traits exist and which classes carry them can change without touching
code; the vector simply grows or shrinks with the number of traits.

Why the *largest* objects: the slot count has to be fixed for the detector,
and in a camera view size is the best cheap proxy for "closest / most
relevant". Unused slots are all zeros, with is_present = 0.0 telling the
model that the rest of the slot is padding rather than an object sitting at
(0, 0) standing still.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from src.implementations.frame_merging_feature_encoder import FrameMergingFeatureEncoder
from src.interfaces.encoder import FeatureVector, StateVector


class BiologicalFeatureEncoder(FrameMergingFeatureEncoder):
    """FrameMergingFeatureEncoder + per-object slots for the N largest objects."""

    #: Per-slot kinematic columns, in vector order. Trait columns follow.
    SLOT_BASE_FIELDS: tuple[str, ...] = ("present", "x", "y", "vx", "vy", "size")

    def __init__(
        self,
        traits: Mapping[str, Iterable[int]],
        max_objects: int = 5,
    ) -> None:
        """Args:
        traits: Trait name -> COCO class ids that carry that trait. Iteration
            order is column order in every slot, so it must be stable
            between training and prediction (a dict or the config value is).
            A class may appear under several traits (multi-hot), or under
            none, in which case its trait columns are all 0.0.
        max_objects: How many object slots to append. Frames with more
            objects keep only the largest ones; frames with fewer get
            zero-filled slots.

        FEATURE_NAMES / FEATURE_DIM are set per instance here rather than on
        the class, because their length depends on these two arguments.
        """
        if max_objects < 1:
            raise ValueError(f"max_objects must be at least 1, got {max_objects}.")

        self._max_objects = max_objects
        self._trait_names: tuple[str, ...] = tuple(traits)
        # Materialise once: membership tests per object per frame should not
        # re-iterate whatever iterable the config handed over.
        self._trait_sets: tuple[frozenset[int], ...] = tuple(
            frozenset(int(class_id) for class_id in traits[name]) for name in self._trait_names
        )
        self._slot_dim = len(self.SLOT_BASE_FIELDS) + len(self._trait_names)

        slot_names = tuple(
            f"obj{slot}_{field}"
            for slot in range(max_objects)
            for field in (
                *self.SLOT_BASE_FIELDS,
                *(f"trait_{name.lower()}" for name in self._trait_names),
            )
        )
        self.FEATURE_NAMES: tuple[str, ...] = FrameMergingFeatureEncoder.FEATURE_NAMES + slot_names
        self.FEATURE_DIM: int = len(self.FEATURE_NAMES)

    @property
    def max_objects(self) -> int:
        return self._max_objects

    @property
    def trait_names(self) -> tuple[str, ...]:
        return self._trait_names

    @property
    def slot_dim(self) -> int:
        """Width of one object slot: 6 kinematic columns + one per trait."""
        return self._slot_dim

    def traits_of(self, object_type: int) -> np.ndarray:
        """Multi-hot trait vector (1.0 / 0.0 per trait, in trait order) for a class id."""
        return np.array(
            [1.0 if object_type in members else 0.0 for members in self._trait_sets],
            dtype=np.float64,
        )

    # --- Internals --------------------------------------------------------------------

    def _encode_group(self, t: int, group: list[StateVector]) -> FeatureVector:
        base = super()._encode_group(t, group)

        # Largest first. Ties are broken by position so the slot order does
        # not depend on the order the tracker happened to report objects in.
        largest = sorted(group, key=lambda s: (-s.size, s.x, s.y))[: self._max_objects]

        # Zeros everywhere is exactly the padding for an empty slot, so only
        # the occupied slots need writing.
        slots = np.zeros((self._max_objects, self._slot_dim), dtype=np.float64)
        n_base = len(self.SLOT_BASE_FIELDS)
        for row, state in zip(slots, largest):
            row[:n_base] = (1.0, state.x, state.y, state.vx, state.vy, state.size)
            row[n_base:] = self.traits_of(state.object_type)

        return FeatureVector(t=t, vector=np.concatenate([base.vector, slots.ravel()]))
