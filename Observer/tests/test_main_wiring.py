"""Tests for the id -> implementation tables in main.py.

The Supervisor and the dashboard only ever see plain id strings; these
tests pin down that those strings, the DetectorKind enum in src.config and
main.DETECTORS all agree.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from main import DETECTORS, FEATURE_ENCODERS, build_detector_factory, build_detectors_provider
from src.config import Config, DetectorKind, FeatureEncoderKind
from src.implementations.autoencoder_detector import PyTorchAutoencoderDetector
from src.implementations.iso_forest_detector import IsolationForestAnomalyDetector


def test_every_kind_is_registered() -> None:
    assert set(DETECTORS) == set(DetectorKind)
    assert set(FEATURE_ENCODERS) == set(FeatureEncoderKind)


def test_kinds_are_their_id_strings() -> None:
    # StrEnum: plain strings from the dashboard compare equal to the members.
    assert DetectorKind.ISOLATION_FOREST == "isolation_forest"
    assert DetectorKind("autoencoder") is DetectorKind.AUTOENCODER


def test_detectors_provider_offers_plain_string_ids() -> None:
    options = build_detectors_provider(Config())()

    assert [o.id for o in options] == ["isolation_forest", "autoencoder"]
    assert all(type(o.id) is str for o in options)


def test_detector_factory_builds_from_plain_strings() -> None:
    factory = build_detector_factory(Config())

    assert isinstance(factory("isolation_forest"), IsolationForestAnomalyDetector)
    assert isinstance(factory("autoencoder"), PyTorchAutoencoderDetector)
    with pytest.raises(ValueError, match="Unknown detector 'nope'"):
        factory("nope")


def test_default_detector_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Config().default_detector is DetectorKind.ISOLATION_FOREST

    monkeypatch.setenv("DEFAULT_DETECTOR", "autoencoder")
    assert Config().default_detector is DetectorKind.AUTOENCODER

    monkeypatch.setenv("DEFAULT_DETECTOR", "isolation-forest")
    with pytest.raises(ValidationError):
        Config()
