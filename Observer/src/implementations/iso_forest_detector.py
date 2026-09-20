"""Implementation 1 of IAnomalyDetector: sklearn IsolationForest.

Design note (confirmed with the project owner): FeatureVector.vector is
already the fixed-size sample the forest trains/predicts.

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

import logging
import math
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.interfaces.detector import IAnomalyDetector, ProgressCallback
from src.interfaces.encoder import FeatureVector

logger = logging.getLogger(__name__)

#: Fewer trees than this and an IsolationForest's score is dominated by the
#: luck of each tree's random splits rather than by the data: on the data the
#: supervisor tests use, a single tree caught the outlier 30% of the time and
#: 100 trees caught it every time. This is also sklearn's own default, and it
#: is the floor whatever coverage asks for - coverage says how many trees are
#: needed to *see* the data, which is a weaker thing to ask.
MIN_TREES = 100


class IsolationForestAnomalyDetector(IAnomalyDetector):
    def __init__(
        self,
        contamination: float | str = 0.05,
        # Percent, strictly between 0 and 100 - 99 means 99%, not 0.99.
        # The unit is in the name on purpose: reading it as a fraction is what
        # once made this forest a single tree.
        n_tree_training_coverage_percent: float = 99.0,
        random_state: int = 42,
    ) -> None:
        self._contamination = contamination
        self._n_tree_training_coverage_percent = n_tree_training_coverage_percent
        self._random_state = random_state
        self._model: IsolationForest | None = None

    @staticmethod
    def _stack(vectors: list[FeatureVector]) -> np.ndarray:
        return np.stack([fv.vector for fv in vectors], axis=0)

    @staticmethod
    def _calculate_n_estimators(
            dataset_size: int,
            max_samples: int = 256,
            coverage_percent: float = 99.0,
    ) -> int:
        """Calculates the number of trees (n_estimators) needed to achieve
        a target dataset coverage percentage for an Isolation Forest.

        A tree trains on S of the N rows, so a given row is missed by one tree
        with probability (1 - S/N) and by all E trees with (1 - S/N)^E.
        Requiring that miss rate to fall below (1 - C) gives

            E >= ln(1 - C) / ln(1 - S/N)

        That is a *lower* bound - it says how many trees are needed to have
        seen the data, not how many are needed for the score to settle down -
        so the result is floored at MIN_TREES.

        Args:
            dataset_size: Total number of rows in training_data (N).
            max_samples: Number of samples per tree (S).
            coverage_percent: Desired coverage as a PERCENTAGE, strictly
                between 0 and 100 (99.0 = 99%). 100 is excluded rather than
                clamped: random subsampling never guarantees every row is
                seen, so full coverage is a limit that needs infinitely many
                trees. 99.99 asks for practically all of it.

        Returns:
            The recommended number of trees, never fewer than MIN_TREES.
        """
        if dataset_size <= 0:
            raise ValueError("dataset_size must be greater than 0.")
        if max_samples <= 0:
            raise ValueError("max_samples must be greater than 0.")
        if not (0 < coverage_percent < 100):
            raise ValueError(
                "coverage_percent is a percentage strictly between 0 and 100. "
                "100 is excluded because full coverage would take infinitely "
                f"many trees - ask for 99.99 instead. Got {coverage_percent!r}."
            )
        if coverage_percent < 1:
            # Legal, but it is also exactly what a fraction looks like, and
            # asking for under 1% coverage is not something anyone means to do.
            # The MIN_TREES floor now hides this instead of collapsing to one
            # tree, so without a word here it would pass entirely unnoticed.
            logger.warning(
                "coverage_percent=%r asks for less than 1%% coverage. If a fraction "
                "was meant, pass %g instead.",
                coverage_percent,
                coverage_percent * 100,
            )

        # Cap max_samples if it exceeds dataset size
        samples_per_tree = min(max_samples, dataset_size)

        # Every tree sees every row, so coverage is satisfied by the first one
        # and the formula below would take ln(0). The floor still applies: here
        # the trees earn their keep by averaging out random splits rather than
        # by reaching more data.
        if samples_per_tree == dataset_size:
            return MIN_TREES

        # Convert percentage to decimal probability (e.g., 99.0 -> 0.99)
        c = coverage_percent / 100.0

        # E = ln(1 - c) / ln(1 - S / N)
        n_estimators = math.log(1 - c) / math.log(1 - (samples_per_tree / dataset_size))

        return max(MIN_TREES, math.ceil(n_estimators))

    def fit(
        self,
        training_data: list[list[FeatureVector]],
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Fit the forest.

        `progress_callback` is accepted for interface parity but only ever
        fires once, at 100: sklearn's fit() is a single opaque call with no
        way to observe how far along it is. A UI therefore sees the training
        bar jump from 0 to done for this detector, which is honest - there is
        no intermediate state to report.
        """
        flattened: list[FeatureVector] = [
            fv for frame in training_data for fv in frame
        ]
        if not flattened:
            raise ValueError("Cannot fit IsolationForestAnomalyDetector on empty training_data.")

        x_train = self._stack(flattened)
        max_samples = min(256, len(x_train)) # 256 is best number of samples per tree!
        trees_number = self._calculate_n_estimators(
            len(x_train), max_samples, self._n_tree_training_coverage_percent
        )
        logger.info(
            "n_estimators=%d for %d samples at %g%% coverage (max_samples=%d, floor=%d).",
            trees_number,
            len(x_train),
            self._n_tree_training_coverage_percent,
            max_samples,
            MIN_TREES,
        )

        logger.info(
            f"Isolation forest samples taken to training = {trees_number * max_samples} from total training size = {len(x_train)} "
            f"_contamination = {self._contamination}"
        )

        model = IsolationForest(
            n_estimators=trees_number,
            max_samples=max_samples,
            contamination=self._contamination,
            random_state=self._random_state,
            n_jobs=-1,
        )
        model.fit(x_train)
        self._model = model

        if progress_callback is not None:
            try:
                progress_callback(100)
            except Exception:
                logger.exception("Training progress_callback failed; continuing.")

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
        if self._model is None:  #TODO: While save/load should not be state loaded as well?
            raise RuntimeError("Cannot save an untrained IsolationForestAnomalyDetector.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        self._model = joblib.load(path)
