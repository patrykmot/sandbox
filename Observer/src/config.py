"""Application configuration.

A single source of truth for runtime configuration, loaded from environment
variables (or a `.env` file) via pydantic-settings, with sensible defaults so
the app can be started with zero configuration for local development.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    default_detector: str = "isolation_forest"
    """Which detector a run trains with until the dashboard says otherwise.
    One of the ids in main.DETECTORS."""

    isolation_forest_contamination: float = 0.05
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

    # --- Feature inspection ----------------------------------------------------------
    feature_csv_enabled: bool = True
    """Dump every produced FeatureVector to CSV for offline inspection."""
    feature_csv_path: str = "feature_vectors.csv"
    """Where that CSV is written. Truncated at the start of every run."""
