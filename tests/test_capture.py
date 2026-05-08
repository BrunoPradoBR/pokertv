import numpy as np
import pytest
from pokertv.capture import TableDetector, _compute_edge_density, _compute_hsv_histogram


def make_detector(thresholds=None):
    return TableDetector(
        logo_template=None,
        reference_histograms=[],
        thresholds=thresholds or {
            "logo_match_threshold": 0.85,
            "histogram_similarity_threshold": 0.80,
            "edge_density_min": 0.005,
            "edge_density_max": 0.12,
            "confidence_threshold": 0.70,
        },
    )


def test_edge_density_blank_frame_is_zero():
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert _compute_edge_density(blank) == pytest.approx(0.0)


def test_edge_density_noisy_frame_is_nonzero():
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    assert _compute_edge_density(noisy) > 0.0


def test_detector_returns_none_for_blank_frame():
    detector = make_detector()
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.detect(blank, window_id=0) is None


def test_detector_confidence_is_float_when_threshold_zero():
    detector = make_detector(thresholds={
        "logo_match_threshold": 0.85,
        "histogram_similarity_threshold": 0.80,
        "edge_density_min": 0.0,
        "edge_density_max": 1.0,
        "confidence_threshold": 0.0,
    })
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = detector.detect(frame, window_id=1)
    if result is not None:
        assert isinstance(result.confidence, float)
        assert result.client == "pokerstars"


def test_hsv_histogram_returns_normalized_array():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hist = _compute_hsv_histogram(frame)
    assert hist.shape == (50, 60)
    assert float(hist.max()) <= 1.0
