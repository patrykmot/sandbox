"""Application configuration.

A single source of truth for runtime configuration, loaded from environment
variables (or a `.env` file) via pydantic-settings, with sensible defaults so
the app can be started with zero configuration for local development.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DetectorKind(StrEnum):
    """Ids of the selectable anomaly detectors (see main.DETECTORS).

    A StrEnum, so each member *is* its id string: it compares equal to the
    plain strings the dashboard and Supervisor pass around, and env vars
    (DEFAULT_DETECTOR=autoencoder) are validated against it at startup.
    """

    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"


class FeatureEncoderKind(StrEnum):
    """Ids of the selectable feature encoders (see main.FEATURE_ENCODERS)."""

    FRAME_MERGING = "frame_merging"
    """Global frame statistics only."""
    BIOLOGICAL = "biological"
    """Global statistics plus per-object slots with semantic traits."""


#: Default trait table for BiologicalFeatureEncoder: trait name -> MS COCO
#: class ids (as YOLO reports them, 0 = person) that carry the trait. A class
#: may carry several traits (multi-hot) or none. Dict order is column order in
#: the feature vector, so reordering or renaming invalidates a trained model.
DEFAULT_OBJECT_TRAITS: dict[str, list[int]] = {
    "BIOLOGICAL": [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "MOTORIZED": [2, 3, 4, 5, 6, 7, 8],
    "RIDEABLE": [1, 2, 3, 4, 5, 6, 7, 8, 17, 20, 30, 31, 36, 37],
    "HEAVY": [2, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23, 57, 59, 60, 69, 72],
    "PORTABLE": [
        24, 25, 26, 27, 28, 29, 32, 33, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 63, 64, 65, 66, 67, 70, 73, 74, 75, 76, 77, 78, 79,
    ],
    "ANCHORED": [9, 10, 11, 12, 61, 71],
    "AERIAL": [4, 14, 29, 32, 33],
    "FURNITURE": [13, 56, 57, 58, 59, 60, 62, 68, 69, 72],
    "WHEELED": [1, 2, 3, 4, 5, 6, 7, 28, 36],
    "HANDHELD": [25, 34, 35, 38, 42, 43, 44, 64, 65, 67, 76, 78, 79],
}


class Config(BaseSettings):
    """Runtime configuration for the Observer system.

    Every field can be overridden via an environment variable of the same
    name (case-insensitive), or via a `.env` file in the working directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Supervisor / data collection -------------------------------------------------
    collection_target_value: int = 1000
    """Number of feature vectors to collect during COLLECTING_DATA before training."""

    # --- Video source --------------------------------------------------------------
    camera_index: int | str = 0
    """Fallback video source when no camera can be enumerated: an OpenCV
    camera index (int) or an RTSP/video stream URL (str). Normally the
    dashboard's camera picker decides instead."""

    camera_probe_max: int = 5
    """How many device indices to try when enumerating cameras. Cameras are
    found by opening indices 0..camera_probe_max-1 and keeping the ones that
    answer - raise it if a camera sits on a higher index."""

    # --- Web dashboard ---------------------------------------------------------------
    server_host: str = "localhost"
    server_port: int = 8000
    dashboard_max_alarms: int = 20
    """How many recent alarm events (with frame snapshots) the panel keeps."""
    dashboard_stream_fps: int = 30
    """Target frame rate of the MJPEG live view."""
    dashboard_jpeg_quality: int = 80

    # --- Video encoder (YOLO) -------------------------------------------------------
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence: float = 0.51
    yolo_tracker: str = "bytetrack.yaml"

    # --- Anomaly detector ------------------------------------------------------------
    default_detector: DetectorKind = DetectorKind.ISOLATION_FOREST
    """Which detector a run trains with until the dashboard says otherwise.
    Any DetectorKind value; anything else fails at startup."""

    isolation_forest_contamination: float = 0.05
    isolation_training_coverage_percent: float = 95.0
    model_save_path: str = "models/isolation_forest.joblib"

    autoencoder_hidden_dim: int = 32
    autoencoder_latent_dim: int = 8
    """Bottleneck width. Narrower forces the model to generalise harder, so
    unusual frames reconstruct worse - but too narrow and normal ones do too."""
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 32
    autoencoder_learning_rate: float = 1e-3
    autoencoder_percentile_threshold: float = 0.99
    """Where the alarm line sits, as a FRACTION: 0.99 means "the error 99% of
    training frames stayed below". Lower it to catch more and alarm more."""

    # --- Feature encoder -------------------------------------------------------------
    feature_encoder: FeatureEncoderKind = FeatureEncoderKind.BIOLOGICAL
    """Which feature encoder a run uses. Any FeatureEncoderKind value; anything
    else fails at startup. Changing it changes the vector width, so a model
    trained with one cannot score the other."""

    biological_max_objects: int = 5
    """BiologicalFeatureEncoder: how many of the largest objects get their own
    slot. Fewer objects are zero-padded, more are dropped (smallest first)."""

    biological_object_traits: dict[str, list[int]] = Field(
        default_factory=lambda: {name: list(ids) for name, ids in DEFAULT_OBJECT_TRAITS.items()}
    )
    """BiologicalFeatureEncoder: trait name -> COCO class ids carrying it (see
    DEFAULT_OBJECT_TRAITS). Override as JSON, e.g.
    BIOLOGICAL_OBJECT_TRAITS='{"BIOLOGICAL": [0, 16], "MOTORIZED": [2, 3]}'.
    Each trait adds one column per object slot."""

    # --- Feature inspection ----------------------------------------------------------
    feature_csv_enabled: bool = True
    """Dump every produced FeatureVector to CSV for offline inspection."""
    feature_csv_path: str = "feature_vectors.csv"
    """Where that CSV is written. Truncated at the start of every run."""
