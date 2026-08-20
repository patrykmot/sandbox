"""Phase I placeholder IFeatureEncoder implementation.

Real feature engineering (capturing multi-object relations, temporal
context, etc.) is deferred to Phase II. For now this simply maps each
StateVector 1:1 onto a FeatureVector so the rest of the pipeline
(collection -> training -> monitoring -> alarms) can be built and tested
end-to-end today.
"""

from __future__ import annotations

import numpy as np

from src.interfaces.encoder import FeatureVector, IFeatureEncoder, StateVector


class DummyFeatureEncoder(IFeatureEncoder):
    """Maps each StateVector directly to a 6-dim FeatureVector.

    vector = [x, y, vx, vy, size, object_type]
    """

    FEATURE_DIM = 6

    def encode(self, states: list[StateVector]) -> list[FeatureVector]:
        return [
            FeatureVector(
                t=state.t,
                vector=np.array(
                    [state.x, state.y, state.vx, state.vy, state.size, float(state.object_type)],
                    dtype=np.float64,
                ),
            )
            for state in states
        ]
