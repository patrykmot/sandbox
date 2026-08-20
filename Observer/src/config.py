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
    """OpenCV camera index (int) or RTSP/video stream URL (str)."""

    # --- Web server (reserved for Phase II) -----------------------------------------
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # --- Video encoder (YOLO) -------------------------------------------------------
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence: float = 0.25
    yolo_tracker: str = "bytetrack.yaml"

    # --- Anomaly detector ------------------------------------------------------------
    isolation_forest_contamination: float = 0.05
    model_save_path: str = "models/isolation_forest.joblib"
