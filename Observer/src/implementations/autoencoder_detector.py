"""Implementation 2 of IAnomalyDetector: a PyTorch autoencoder.

How it decides
--------------
An autoencoder is trained to squeeze each feature vector through a narrow
bottleneck and rebuild it. Trained only on normal scenes, it becomes good at
rebuilding normal scenes and bad at rebuilding anything else - so the
*reconstruction error* is the anomaly score, and the question is only where
to put the line.

That line is learned, not guessed: after training, every training sample is
run back through the model and `maximum_normal_error` is taken as a high
percentile of the resulting errors. The percentile (rather than the maximum)
is what keeps a handful of odd training frames from setting the bar so high
that nothing ever trips it.

    is_anomaly = output_error > maximum_normal_error

Standardization
---------------
The features are not on comparable scales - FrameMergingFeatureEncoder emits
object counts in the single digits next to NO_PAIR_DISTANCE = 7_777_777.
Squared error over raw columns would be almost entirely the distance columns,
and the model would be blind to speed and size. So mean/std are fitted per
column at training time, stored with the weights, and applied identically in
predict(). IsolationForest needs none of this - a tree splits per column and
does not care about scale - which is why only this detector carries it.

Frame-level result
------------------
Like IsolationForestAnomalyDetector, this scores each FeatureVector in a
frame on its own and combines them: the frame's error is the *worst* (max)
of them, matching the forest's "any object anomalous -> the frame is". If
that strategy needs to change, this is the only place to edit.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch
from torch import nn

from src.interfaces.detector import IAnomalyDetector, ProgressCallback
from src.interfaces.encoder import FeatureVector

logger = logging.getLogger(__name__)

#: Standard deviation below which a column is treated as constant. Dividing
#: by its real (~0) std would produce inf/nan, so such columns are left
#: centred but unscaled.
_MIN_STD = 1e-8


class AutoencoderNet(nn.Module):
    """A symmetric dense autoencoder: input -> hidden -> latent -> hidden -> input.

    The output layer is deliberately linear: the input has been standardized,
    so reconstructions are signed and unbounded and any squashing activation
    here would cap what the model can express.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PyTorchAutoencoderDetector(IAnomalyDetector):
    """Flags a frame whose reconstruction error exceeds the learned baseline."""

    def __init__(
        self,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        percentile_threshold: float = 0.99,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            percentile_threshold: Where to draw the line, as a FRACTION
                between 0 and 1 (0.95-0.99 is the usual range). 0.99 means
                "the error 99% of training samples stayed below". Passing a
                0-100 percentile here is a common slip and raises rather than
                silently thresholding near the minimum, which would make
                every frame an anomaly.
            random_state: Seeds torch so a given training set gives the same
                model twice.
        """
        if not 0.0 < percentile_threshold <= 1.0:
            raise ValueError(
                "percentile_threshold is a fraction in (0, 1] - e.g. 0.99 for the "
                f"99th percentile, not 99. Got {percentile_threshold!r}."
            )

        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._percentile_threshold = percentile_threshold
        self._random_state = random_state

        # CPU on purpose: this targets an edge box with no guaranteed GPU, and
        # a silent .cuda() would be a surprise rather than a feature.
        self._device = torch.device("cpu")

        self._model: AutoencoderNet | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self.maximum_normal_error: float | None = None

    # --- Public state -------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self._model is not None and self.maximum_normal_error is not None

    # --- Training -----------------------------------------------------------------------

    def fit(
        self,
        training_data: list[list[FeatureVector]],
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        flattened: list[FeatureVector] = [fv for frame in training_data for fv in frame]
        if not flattened:
            raise ValueError("Cannot fit PyTorchAutoencoderDetector on empty training_data.")

        torch.manual_seed(self._random_state)

        x_raw = np.stack([fv.vector for fv in flattened], axis=0).astype(np.float32)
        n_samples, n_features = x_raw.shape
        logger.info(
            "Training autoencoder on %d samples x %d features "
            "(epochs=%d, batch=%d, latent=%d).",
            n_samples,
            n_features,
            self._epochs,
            self._batch_size,
            self._latent_dim,
        )

        x = self._fit_standardizer(x_raw)
        model = AutoencoderNet(n_features, self._hidden_dim, self._latent_dim).to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        error_function = nn.MSELoss()

        model.train()
        for epoch in range(self._epochs):
            # Reshuffled every epoch so batches are not the same partition
            # each time - otherwise the model can learn the batching.
            order = torch.randperm(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, self._batch_size):
                batch = x[order[start : start + self._batch_size]]
                optimizer.zero_grad()
                loss = error_function(model(batch), batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            epoch_loss /= n_samples
            percent = int(round((epoch + 1) / self._epochs * 100))
            logger.info("Epoch %d/%d - loss %.6f", epoch + 1, self._epochs, epoch_loss)
            self._print_progress(percent, final=False)
            self._report(progress_callback, percent)

        self._model = model
        self.maximum_normal_error = self._compute_threshold(x)
        self._print_progress(100, final=True)
        self._report(progress_callback, 100)
        logger.info(
            "Autoencoder trained. maximum_normal_error = %.6f (%.0fth percentile).",
            self.maximum_normal_error,
            self._percentile_threshold * 100,
        )

    def _fit_standardizer(self, x_raw: np.ndarray) -> torch.Tensor:
        """Learn per-column mean/std, keep them, and return the scaled tensor."""
        mean = x_raw.mean(axis=0)
        std = x_raw.std(axis=0)
        # A constant column has std ~0; scaling by 1.0 leaves it centred at
        # zero and contributing nothing, which is exactly right.
        std = np.where(std < _MIN_STD, 1.0, std)

        self._mean = torch.from_numpy(mean.astype(np.float32)).to(self._device)
        self._std = torch.from_numpy(std.astype(np.float32)).to(self._device)
        return self._standardize(torch.from_numpy(x_raw).to(self._device))

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        return (x - self._mean) / self._std

    def _compute_threshold(self, x: torch.Tensor) -> float:
        """The error the training set stayed below, at the configured percentile."""
        error_per_sample = self._errors(x).numpy()
        return float(
            np.percentile(error_per_sample, self._percentile_threshold * 100)
        )

    def _errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error for already-scaled input."""
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            reconstructed = self._model(x)
            return ((reconstructed - x) ** 2).mean(dim=1)

    @staticmethod
    def _report(progress_callback: ProgressCallback | None, percent: int) -> None:
        """Push progress outward. A UI that throws must not abort training."""
        if progress_callback is None:
            return
        try:
            progress_callback(percent)
        except Exception:
            logger.exception("Training progress_callback failed; continuing.")

    @staticmethod
    def _print_progress(percent: int, final: bool, bar_width: int = 20) -> None:
        """One in-place-updating console line, matching the collection bar.

        A bare carriage return rather than logging: log lines carry a
        timestamp prefix and start a new line each call, which would scroll
        one line per epoch instead of redrawing one bar.
        """
        filled = int(bar_width * percent / 100)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\rTraining anomaly detector:  [{bar}] {percent:3d}%")
        sys.stdout.write("\n" if final else "")
        sys.stdout.flush()

    # --- Detection ----------------------------------------------------------------------

    def predict(self, vector: list[FeatureVector]) -> tuple[bool, float]:
        if not self.is_trained():
            raise RuntimeError(
                "PyTorchAutoencoderDetector.predict() called before fit()/load()."
            )
        if not vector:
            # Nothing detected this frame - nothing to flag as anomalous.
            return False, 0.0

        x_raw = np.stack([fv.vector for fv in vector], axis=0).astype(np.float32)
        x = self._standardize(torch.from_numpy(x_raw).to(self._device))

        # The worst object decides the frame, as it does for the forest.
        output_error = float(self._errors(x).max().item())

        assert self.maximum_normal_error is not None
        is_anomaly = output_error > self.maximum_normal_error
        return is_anomaly, output_error

    # --- Persistence ---------------------------------------------------------------------
    #
    # Only tensors, ints and floats go into the checkpoint, so it loads back
    # under torch.load(weights_only=True) - no pickle escape hatch needed to
    # read a model file.

    def save(self, path: str) -> None:
        if not self.is_trained():
            raise RuntimeError("Cannot save an untrained PyTorchAutoencoderDetector.")
        assert self._model is not None and self._mean is not None and self._std is not None

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "mean": self._mean,
                "std": self._std,
                "maximum_normal_error": float(self.maximum_normal_error),
                "input_dim": int(self._mean.numel()),
                "hidden_dim": int(self._hidden_dim),
                "latent_dim": int(self._latent_dim),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)

        self._hidden_dim = int(checkpoint["hidden_dim"])
        self._latent_dim = int(checkpoint["latent_dim"])
        model = AutoencoderNet(
            int(checkpoint["input_dim"]), self._hidden_dim, self._latent_dim
        ).to(self._device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        self._model = model
        self._mean = checkpoint["mean"].to(self._device)
        self._std = checkpoint["std"].to(self._device)
        self.maximum_normal_error = float(checkpoint["maximum_normal_error"])
