"""Interface 3: Anomaly Detector."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from src.interfaces.encoder import FeatureVector

#: Signature of the optional training-progress hook. Receives whole percent,
#: 0 to 100, and is called from the training thread.
ProgressCallback = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class DetectorOption:
    """One selectable detector implementation, as offered to the operator.

    The mirror of CameraOption (src/interfaces/video_source.py): `id` is what
    a selection request carries, `name` is what the dropdown shows.
    """

    id: str
    name: str


class IAnomalyDetector(ABC):
    """Abstract interface for training and predicting anomalies from feature vectors."""

    @abstractmethod
    def fit(
        self,
        training_data: list[list[FeatureVector]],
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Train the underlying model on collected vectors.

        Args:
            training_data: A list of frames, each frame being the list of
                FeatureVectors observed at that point in time (0..N objects).
            progress_callback: Optional hook called with whole percent (0-100)
                as training advances, so a UI can show a progress bar. An
                implementation whose training is one opaque call has nothing
                to report partway and may simply call it once with 100 when
                it finishes. Implementations must not let a raising callback
                abort training.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, vector: list[FeatureVector]) -> tuple[bool, float]:
        """Predict whether the current frame's objects are anomalous.

        Args:
            vector: The list of FeatureVectors for all objects observed in
                the current frame.

        Returns:
            (is_anomaly, anomaly_score). The scale and sign of the score are
            the implementation's own - compare scores within one detector,
            never across two.
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
