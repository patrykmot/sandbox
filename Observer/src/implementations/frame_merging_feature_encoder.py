"""IFeatureEncoder implementation that merges a whole frame into one vector.

Where DummyFeatureEncoder emits one FeatureVector per object (so the model
only ever sees objects in isolation), this encoder groups StateVectors by
timestamp and describes each frame as a whole: how many objects there are,
how spread out they are, how fast they move, and how fast they move relative
to each other.

That is what makes multi-object *relations* learnable - "three people
drifting apart slowly" and "three people converging fast" produce very
different vectors even though the individual objects look unremarkable.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.interfaces.encoder import FeatureVector, IFeatureEncoder, StateVector


class FrameMergingFeatureEncoder(IFeatureEncoder):
    """Groups StateVectors by timestamp; emits one FeatureVector per group."""

    #: Column order of FeatureVector.vector. Exposed so tests, debugging and
    #: any future model-inspection code can label the numbers.
    FEATURE_NAMES: tuple[str, ...] = (
        "num_objects",
        "dist_min",
        "dist_p10",
        "dist_p50",
        "dist_p90",
        "rel_speed_p10",
        "rel_speed_p50",
        "rel_speed_p90",
        "rel_speed_max",
        "speed_p10",
        "speed_p50",
        "speed_p90",
        "speed_max",
        "size_p10",
        "size_p50",
        "size_p90",
        "size_max",
    )

    FEATURE_DIM: int = len(FEATURE_NAMES)

    def encode(self, states: list[StateVector]) -> list[FeatureVector]:
        """Merge each same-timestamp group of objects into a single FeatureVector.

        The output list is ordered by timestamp ascending, regardless of the
        input order, so the result is deterministic. In the live pipeline
        every StateVector in a call shares one timestamp, so this returns a
        single FeatureVector per frame; grouping matters for batch/offline
        use where several frames arrive together.
        """
        groups: dict[int, list[StateVector]] = defaultdict(list)
        for state in states:
            groups[state.t].append(state)

        return [self._encode_group(t, groups[t]) for t in sorted(groups)]

    # --- Internals --------------------------------------------------------------------

    def _encode_group(self, t: int, group: list[StateVector]) -> FeatureVector:
        positions = np.array([[s.x, s.y] for s in group], dtype=np.float64)
        velocities = np.array([[s.vx, s.vy] for s in group], dtype=np.float64)
        sizes = np.array([s.size for s in group], dtype=np.float64)

        num_objects = len(group)

        # Absolute speed of each object: ||(vx, vy)||.
        speeds = np.linalg.norm(velocities, axis=1)

        # Pairwise stats need at least two objects. With a single object
        # there are no pairs at all, so these are reported as 0.0 rather
        # than NaN: IsolationForest rejects non-finite input, and no
        # information is actually lost - num_objects is itself a feature, so
        # the model can learn that "num_objects == 1" implies the pairwise
        # columns are structurally zero rather than genuinely measured.
        if num_objects >= 2:
            upper = np.triu_indices(num_objects, k=1)
            pair_distances = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=-1
            )[upper]
            pair_rel_speeds = np.linalg.norm(
                velocities[:, None, :] - velocities[None, :, :], axis=-1
            )[upper]
        else:
            pair_distances = np.empty(0, dtype=np.float64)
            pair_rel_speeds = np.empty(0, dtype=np.float64)

        dist_p10, dist_p50, dist_p90 = self._percentiles(pair_distances)
        rel_p10, rel_p50, rel_p90 = self._percentiles(pair_rel_speeds)
        speed_p10, speed_p50, speed_p90 = self._percentiles(speeds)
        size_p10, size_p50, size_p90 = self._percentiles(sizes)

        vector = np.array(
            [
                float(num_objects),
                self._min(pair_distances),
                dist_p10,
                dist_p50,
                dist_p90,
                rel_p10,
                rel_p50,
                rel_p90,
                self._max(pair_rel_speeds),
                speed_p10,
                speed_p50,
                speed_p90,
                self._max(speeds),
                size_p10,
                size_p50,
                size_p90,
                self._max(sizes),
            ],
            dtype=np.float64,
        )

        return FeatureVector(t=t, vector=vector)

    @staticmethod
    def _percentiles(values: np.ndarray) -> tuple[float, float, float]:
        """10th/50th/90th percentiles, or zeros when there is nothing to measure."""
        if values.size == 0:
            return 0.0, 0.0, 0.0
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        return float(p10), float(p50), float(p90)

    @staticmethod
    def _min(values: np.ndarray) -> float:
        return float(values.min()) if values.size else 0.0

    @staticmethod
    def _max(values: np.ndarray) -> float:
        return float(values.max()) if values.size else 0.0
