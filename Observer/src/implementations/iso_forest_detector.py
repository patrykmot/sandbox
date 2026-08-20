"""Implementation 1 of IAnomalyDetector: sklearn IsolationForest.

Design note (confirmed with the project owner): FeatureVector.vector is
already the fixed-size sample the forest trains/predicts on - this class does
NOT aggregate a frame's multiple objects into one combined vector itself.

  * fit(): every FeatureVector across every frame in `training_data` is
    flattened into one (n_samples, n_features) matrix and used to fit a
    single IsolationForest.
  * predict(): the current frame's FeatureVectors are stacked into a
    (n_objects, n_features) matrix and scored individually. The frame-level
    result is then combined as:
      - is_anomaly = True if ANY object in the frame is flagged anomalous.
      - anomaly_score = the minimum (i.e. most anomalous / worst) of the
        per-object scores.
    If this combination strategy needs to change (e.g. mean instead of min),
    this is the only place to edit.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.interfaces.detector import IAnomalyDetector
from src.interfaces.encoder import FeatureVector


class IsolationForestAnomalyDetector(IAnomalyDetector):
    def __init__(
        self,
        contamination: float | str = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._model: IsolationForest | None = None

    @staticmethod
    def _stack(vectors: list[FeatureVector]) -> np.ndarray:
        return np.stack([fv.vector for fv in vectors], axis=0)

    def fit(self, training_data: list[list[FeatureVector]]) -> None:
        flattened: list[FeatureVector] = [
            fv for frame in training_data for fv in frame
        ]
        if not flattened:
            raise ValueError("Cannot fit IsolationForestAnomalyDetector on empty training_data.")

        x_train = self._stack(flattened)
        model = IsolationForest(
            n_estimators=self._n_estimators,
            max_samples=min(256, len(x_train)),
            contamination=self._contamination,
            random_state=self._random_state,
            n_jobs=-1,
        )
        model.fit(x_train)
        self._model = model

    def predict(self, vector: list[FeatureVector]) -> tuple[bool, float]:
        if self._model is None:
            raise RuntimeError("IsolationForestAnomalyDetector.predict() called before fit()/load().")
        if not vector:
            # Nothing detected this frame - nothing to flag as anomalous.
            return False, 0.0

        x = self._stack(vector)
        predictions = self._model.predict(x)  # +1 normal, -1 anomaly
        scores = self._model.score_samples(x)  # more negative = more anomalous

        is_anomaly = bool(np.any(predictions == -1))
        anomaly_score = float(np.min(scores))
        return is_anomaly, anomaly_score

    def is_trained(self) -> bool:
        return self._model is not None

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an untrained IsolationForestAnomalyDetector.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        self._model = joblib.load(path)
