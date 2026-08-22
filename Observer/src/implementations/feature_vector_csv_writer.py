"""Writes every produced FeatureVector to a CSV file for offline inspection.

Purely an observation aid: it sits alongside the pipeline and never feeds
anything back into it. Enable it with FEATURE_CSV_ENABLED in the config.

The header is written lazily, on the first row, so the column count always
matches the data actually being produced. Column names come from the
encoder's FEATURE_NAMES when it publishes one (FrameMergingFeatureEncoder
does); otherwise they fall back to f0, f1, ... so any IFeatureEncoder
implementation still yields a readable file.
"""

from __future__ import annotations

import csv
import logging
import os

from src.interfaces.encoder import FeatureVector

logger = logging.getLogger(__name__)


class FeatureVectorCsvWriter:
    """Appends FeatureVectors to a CSV file, one row per vector."""

    def __init__(self, path: str, feature_names: tuple[str, ...] | None = None) -> None:
        """Args:
        path: Destination CSV path. Truncated on construction so each run
            starts a fresh file rather than appending to the previous run.
        feature_names: Column names for FeatureVector.vector. When None,
            generic f0..fN names are generated from the first row's width.
        """
        self._path = path
        self._feature_names = feature_names
        self._header_written = False

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # newline="" is required by the csv module to avoid blank rows on Windows.
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

    @property
    def path(self) -> str:
        return self._path

    def write(self, features: list[FeatureVector]) -> None:
        """Write one CSV row per FeatureVector. Safe to call with an empty list."""
        if not features:
            return

        if not self._header_written:
            self._write_header(len(features[0].vector))

        for feature in features:
            self._writer.writerow([feature.t, *feature.vector.tolist()])

        # Flushed per call so the file can be tailed/opened while the system
        # is still running - the whole point is watching the data arrive.
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    # --- Internals --------------------------------------------------------------------

    def _write_header(self, vector_width: int) -> None:
        names = self._feature_names
        if names is None or len(names) != vector_width:
            if names is not None:
                logger.warning(
                    "Encoder reported %d feature names but vectors are %d wide; "
                    "falling back to generic column names.",
                    len(names),
                    vector_width,
                )
            names = tuple(f"f{i}" for i in range(vector_width))

        self._writer.writerow(["t", *names])
        self._header_written = True

    def __enter__(self) -> FeatureVectorCsvWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
