"""Tests for the FeatureVector CSV observation side channel."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.implementations.feature_vector_csv_writer import FeatureVectorCsvWriter
from src.implementations.frame_merging_feature_encoder import FrameMergingFeatureEncoder
from src.interfaces.encoder import FeatureVector, StateVector

T = 1_700_000_000_000


def read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_writes_header_from_encoder_feature_names(tmp_path: Path) -> None:
    path = tmp_path / "feature_vectors.csv"
    writer = FeatureVectorCsvWriter(
        str(path), feature_names=FrameMergingFeatureEncoder.FEATURE_NAMES
    )

    writer.write([FeatureVector(t=T, vector=np.arange(17, dtype=np.float64))])
    writer.close()

    rows = read_csv(path)
    assert rows[0] == ["t", *FrameMergingFeatureEncoder.FEATURE_NAMES]
    assert rows[1][0] == str(T)
    assert len(rows[1]) == 18  # t + 17 features


def test_falls_back_to_generic_column_names(tmp_path: Path) -> None:
    """Any IFeatureEncoder still produces a readable file, names or not."""
    path = tmp_path / "feature_vectors.csv"
    writer = FeatureVectorCsvWriter(str(path))

    writer.write([FeatureVector(t=T, vector=np.array([1.0, 2.0, 3.0]))])
    writer.close()

    assert read_csv(path)[0] == ["t", "f0", "f1", "f2"]


def test_mismatched_name_count_falls_back_rather_than_mislabelling(tmp_path: Path) -> None:
    """Wrong-width names would silently mislabel columns - generic names are safer."""
    path = tmp_path / "feature_vectors.csv"
    writer = FeatureVectorCsvWriter(str(path), feature_names=("only", "two"))

    writer.write([FeatureVector(t=T, vector=np.array([1.0, 2.0, 3.0]))])
    writer.close()

    assert read_csv(path)[0] == ["t", "f0", "f1", "f2"]


def test_one_row_per_feature_vector_with_single_header(tmp_path: Path) -> None:
    path = tmp_path / "feature_vectors.csv"
    writer = FeatureVectorCsvWriter(str(path), feature_names=("a", "b"))

    # Three separate write() calls, mimicking three frames.
    writer.write([FeatureVector(t=T, vector=np.array([1.0, 2.0]))])
    writer.write([FeatureVector(t=T + 33, vector=np.array([3.0, 4.0]))])
    writer.write(
        [
            FeatureVector(t=T + 66, vector=np.array([5.0, 6.0])),
            FeatureVector(t=T + 66, vector=np.array([7.0, 8.0])),
        ]
    )
    writer.close()

    rows = read_csv(path)
    assert rows[0] == ["t", "a", "b"]
    assert len(rows) == 5  # header + 4 vectors
    assert [r[1:] for r in rows[1:]] == [
        ["1.0", "2.0"],
        ["3.0", "4.0"],
        ["5.0", "6.0"],
        ["7.0", "8.0"],
    ]


def test_empty_write_produces_no_header_or_rows(tmp_path: Path) -> None:
    """Frames with no detections must not emit blank rows."""
    path = tmp_path / "feature_vectors.csv"
    writer = FeatureVectorCsvWriter(str(path), feature_names=("a",))

    writer.write([])
    writer.close()

    assert read_csv(path) == []


def test_data_is_readable_while_still_open(tmp_path: Path) -> None:
    """Each write flushes, so the file can be tailed during a live run."""
    path = tmp_path / "feature_vectors.csv"
    writer = FeatureVectorCsvWriter(str(path), feature_names=("a",))

    writer.write([FeatureVector(t=T, vector=np.array([42.0]))])

    rows = read_csv(path)  # deliberately read before close()
    assert rows == [["t", "a"], [str(T), "42.0"]]

    writer.close()


def test_truncates_previous_run(tmp_path: Path) -> None:
    path = tmp_path / "feature_vectors.csv"
    path.write_text("stale,data\n1,2\n", encoding="utf-8")

    writer = FeatureVectorCsvWriter(str(path), feature_names=("a",))
    writer.write([FeatureVector(t=T, vector=np.array([1.0]))])
    writer.close()

    rows = read_csv(path)
    assert rows[0] == ["t", "a"]
    assert len(rows) == 2


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "feature_vectors.csv"

    writer = FeatureVectorCsvWriter(str(path), feature_names=("a",))
    writer.write([FeatureVector(t=T, vector=np.array([1.0]))])
    writer.close()

    assert path.exists()


def test_end_to_end_with_real_encoder(tmp_path: Path) -> None:
    """The realistic path: encoder output straight into the CSV."""
    path = tmp_path / "feature_vectors.csv"
    encoder = FrameMergingFeatureEncoder()

    with FeatureVectorCsvWriter(str(path), feature_names=encoder.FEATURE_NAMES) as writer:
        writer.write(
            encoder.encode(
                [
                    StateVector(t=T, x=0.0, y=0.0, vx=0.0, vy=0.0, size=100.0, object_type=0),
                    StateVector(t=T, x=3.0, y=4.0, vx=3.0, vy=4.0, size=200.0, object_type=2),
                ]
            )
        )

    rows = read_csv(path)
    header, row = rows[0], rows[1]
    assert header == ["t", *FrameMergingFeatureEncoder.FEATURE_NAMES]

    values = dict(zip(header, row))
    assert values["t"] == str(T)
    assert float(values["num_objects"]) == 2.0
    assert float(values["dist_min"]) == 5.0  # 3-4-5 triangle
    assert float(values["size_max"]) == 200.0
