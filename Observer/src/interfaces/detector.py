"""Interface 3: Anomaly Detector."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.interfaces.encoder import FeatureVector


class IAnomalyDetector(ABC):
    """Abstract interface for training and predicting anomalies from feature vectors."""

    @abstractmethod
    def fit(self, training_data: list[list[FeatureVector]]) -> None:
        """Train the underlying model on collected vectors.

        Args:
            training_data: A list of frames, each frame being the list of
                FeatureVectors observed at that point in time (0..N objects).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, vector: list[FeatureVector]) -> tuple[bool, float]:
        """Predict whether the current frame's objects are anomalous.

        Args:
            vector: The list of FeatureVectors for all objects observed in
                the current frame.

        Returns:
            (is_anomaly, anomaly_score).
        """
        raise NotImplementedError

    @abstractmethod
    def is_trained(self) -> bool:
        """Return whether the model has been fitted and is ready to predict."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously persisted model from disk."""
        raise NotImplementedError
