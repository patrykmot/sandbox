"""Entry point: wires the default implementations together and runs the system.

To swap any component for a different implementation (e.g. a video-file
source instead of a live camera, or a different anomaly detector), only the
imports and constructor calls below need to change - every other module
depends solely on the interfaces in src/interfaces/, so nothing else in the
system needs to know.

Threading model (Phase II requirement): the OpenCV/Supervisor processing loop
runs on a background daemon thread so it never blocks FastAPI's event loop,
while the dashboard serves requests on the main thread. The dashboard pulls
what it displays from the Supervisor and receives alarms as one more
IAlarmHandler sink - the Supervisor itself knows nothing about a UI.

The system boots IDLE: the selected camera is previewed, but nothing is
detected or learned until START is pressed in the dashboard.
"""

from __future__ import annotations

import logging
import threading

from src.config import Config
from src.core.supervisor import Supervisor
from src.implementations.autoencoder_detector import PyTorchAutoencoderDetector
from src.implementations.camera_list import list_cameras
from src.implementations.camera_source import CameraVideoSource
from src.implementations.composite_alarm import CompositeAlarmHandler
from src.implementations.console_alarm import ConsoleLoggerAlarmHandler
from src.implementations.feature_vector_csv_writer import FeatureVectorCsvWriter
from src.implementations.frame_merging_feature_encoder import FrameMergingFeatureEncoder
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector
from src.implementations.web_controller import WebController
from src.implementations.yolo_encoder import YOLOVideoEncoder
from src.interfaces.alarm import IAlarmHandler
from src.interfaces.detector import DetectorOption, IAnomalyDetector
from src.interfaces.video_source import CameraOption, IVideoSource

logger = logging.getLogger(__name__)


# Component 3: the selectable Anomaly Detectors. This table is the single
# place that maps a dashboard dropdown entry to a constructor - add an
# implementation of IAnomalyDetector here and it appears in the UI. The
# operator picks one per run, so these are built on START rather than now.
DETECTORS: dict[str, str] = {
    "isolation_forest": "Isolation Forest (scikit-learn)",
    "autoencoder": "Autoencoder (PyTorch)",
}


def build_detectors_provider(config: Config):
    """Returns the callable the dashboard uses to fill its detector dropdown."""

    def provider() -> list[DetectorOption]:
        return [DetectorOption(id=key, name=name) for key, name in DETECTORS.items()]

    return provider


def build_detector_factory(config: Config):
    """Returns the callable the Supervisor uses to build the selected detector.

    An unknown id raises rather than quietly falling back: the Supervisor
    reports it and stays IDLE, which is visible, where a silent substitution
    would have the dashboard claim one detector while another ran.
    """

    def factory(detector: str) -> IAnomalyDetector:
        if detector == "isolation_forest":
            return IsolationForestAnomalyDetector(
                contamination=config.isolation_forest_contamination,
                n_tree_training_coverage_percent=config.isolation_training_coverage_percent,
            )
        if detector == "autoencoder":
            return PyTorchAutoencoderDetector(
                hidden_dim=config.autoencoder_hidden_dim,
                latent_dim=config.autoencoder_latent_dim,
                epochs=config.autoencoder_epochs,
                batch_size=config.autoencoder_batch_size,
                learning_rate=config.autoencoder_learning_rate,
                percentile_threshold=config.autoencoder_percentile_threshold,
            )
        raise ValueError(f"Unknown detector {detector!r}. Known: {sorted(DETECTORS)}.")

    return factory


def build_cameras_provider(config: Config):
    """Returns the callable the dashboard uses to fill its camera dropdown."""

    def provider(active: int | str | None = None) -> list[CameraOption]:
        return list_cameras(max_probe=config.camera_probe_max, always_include=active)

    return provider


def default_camera(config: Config) -> int | str:
    """The camera selected on boot: the first one found, else the configured one."""
    cameras = list_cameras(max_probe=config.camera_probe_max)
    if cameras:
        return cameras[0].id
    logger.warning("No camera found; falling back to configured CAMERA_INDEX=%r.", config.camera_index)
    return config.camera_index


def build_supervisor(config: Config, alarm_handler: IAlarmHandler) -> Supervisor:
    # Component 1: Video Data Source. The operator picks the camera at
    # runtime, so the Supervisor gets a factory and opens sources itself.
    # Swap for e.g. a VideoFileSource by changing this single line.
    def video_source_factory(camera: int | str) -> IVideoSource:
        return CameraVideoSource(camera_index=camera)

    # Component 2: Video Encoder. Swap for another detector/tracker backend
    # by implementing IVideoEncoder and constructing it here instead.
    video_encoder = YOLOVideoEncoder(
        model_path=config.yolo_model_path,
        confidence=config.yolo_confidence,
        tracker=config.yolo_tracker,
    )

    # Component 6: Feature Encoder. FrameMergingFeatureEncoder merges all
    # objects sharing a timestamp into ONE 17-feature vector per frame, so
    # object-to-object relations are what the detector learns. Swap in
    # DummyFeatureEncoder here for one vector per object instead.
    feature_encoder = FrameMergingFeatureEncoder()

    # Optional observation side channel: dumps every FeatureVector to CSV.
    # Column names are taken from the encoder when it publishes them.
    feature_csv_writer = None
    if config.feature_csv_enabled:
        feature_csv_writer = FeatureVectorCsvWriter(
            path=config.feature_csv_path,
            feature_names=getattr(feature_encoder, "FEATURE_NAMES", None),
        )
        logger.info("Will save FeatureVectors to %s", feature_csv_writer.path)

    return Supervisor(
        video_source_factory=video_source_factory,
        video_encoder=video_encoder,
        feature_encoder=feature_encoder,
        anomaly_detector_factory=build_detector_factory(config),
        alarm_handler=alarm_handler,
        collection_target_value=config.collection_target_value,
        feature_csv_writer=feature_csv_writer,
        camera=default_camera(config),
        detector=config.default_detector,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = Config()

    # Component 7: the operator UI. Swap for a TUI or a metrics exporter by
    # writing one that pulls ISupervisorPort and constructing it here instead.
    controller = WebController(
        host=config.server_host,
        port=config.server_port,
        max_alarms=config.dashboard_max_alarms,
        stream_fps=config.dashboard_stream_fps,
        jpeg_quality=config.dashboard_jpeg_quality,
        cameras_provider=build_cameras_provider(config),
        detectors_provider=build_detectors_provider(config),
    )

    # Component 5: Alarm Handlers. An alarm has more than one audience, and
    # the composite is what keeps one failing sink from taking down the loop.
    # Add an email/SMS/webhook sink by implementing IAlarmHandler and
    # appending it to this list.
    alarm_handler = CompositeAlarmHandler(
        [ConsoleLoggerAlarmHandler(), controller]
    )

    supervisor = build_supervisor(config, alarm_handler)

    # The dashboard reads and drives the Supervisor through this, and only
    # through this: reads are lock-guarded snapshots, and requests are queued
    # and applied on the loop's own thread.
    controller.bind_supervisor(supervisor)

    # The processing loop MUST NOT run on the main thread - FastAPI's event
    # loop lives there. daemon=True so Ctrl+C on the server tears it down.
    processing_thread = threading.Thread(
        target=supervisor.run_forever,
        name="supervisor-loop",
        daemon=True,
    )
    processing_thread.start()

    logger.info(
        "Observer ready on http://%s:%d/ - previewing camera %r, detector %r, "
        "press Start to collect %d frames.",
        config.server_host,
        config.server_port,
        supervisor.camera,
        supervisor.detector,
        config.collection_target_value,
    )

    # Blocks until shutdown.
    controller.run()


if __name__ == "__main__":
    main()
