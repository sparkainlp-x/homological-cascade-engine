"""Regression tests for v0.6.0 robust telemetry and POT thresholds."""

import numpy as np
import pytest

from src.oes32_robust import RobustMADDetector
from src.pot_threshold import POTThreshold


def test_mad_huber_detector_handles_student_t_noise_and_detects_spike():
    rng = np.random.default_rng(3)
    data = rng.standard_t(df=3, size=(2500, 8))
    data[100, 0] = 35.0
    detector = RobustMADDetector(threshold=6.0)
    scores = detector.score(data)
    detections = detector.detect(data)
    assert scores.shape == (2500, 8)
    assert scores[100, 0] > 6.0
    assert detections[100, 0]
    assert abs(np.median(data[:, 0])) < 0.15


def test_mask_aware_missing_data_and_axis_handling():
    rng = np.random.default_rng(8)
    data = rng.normal(size=(100, 4))
    data[12, 0] = 25.0
    data[4, 1] = np.nan
    mask = np.ones_like(data, dtype=bool)
    mask[12, 0] = False
    mask[6, 2] = False
    data[6, 2] = 1e8
    detector = RobustMADDetector(threshold=5.0)
    detections = detector.detect(data, mask=mask, axis=0)
    assert not detections[12, 0]
    assert not detections[4, 1]
    assert not detections[6, 2]
    transposed_scores = detector.score(data.T, axis=1)
    np.testing.assert_allclose(transposed_scores, detector.score(data, axis=0).T)


def test_top_k_keeps_only_strongest_candidates_in_each_block():
    rng = np.random.default_rng(9)
    data = rng.normal(scale=0.2, size=(500, 4))
    data[250, :] = [20.0, 18.0, 16.0, 14.0]
    detector = RobustMADDetector(threshold=4.0, top_k=1, block_size=2)
    detected = detector.detect(data)
    assert detected[250, 0]
    assert not detected[250, 1]
    assert detected[250, 2]
    assert not detected[250, 3]


def test_detector_validation_and_all_missing_feature():
    detector = RobustMADDetector()
    with pytest.raises(ValueError):
        detector.score(np.ones(3), axis=2)
    with pytest.raises(ValueError):
        detector.score(np.ones((4, 2)), mask=np.ones(4))
    with pytest.raises(ValueError):
        RobustMADDetector(top_k=1, block_size=0)
    missing = np.full((20, 2), np.nan)
    missing[:, 0] = 1.0
    assert not detector.detect(missing)[:, 1].any()


def test_pot_fit_stream_update_and_anomaly_masking():
    rng = np.random.default_rng(14)
    calibration = rng.pareto(a=2.5, size=3000)
    estimator = POTThreshold(tail_quantile=0.9, target_rate=0.01, min_exceedances=10)
    estimator.fit(calibration)
    original_threshold = estimator.threshold
    batch = np.concatenate((rng.pareto(a=2.5, size=80), [10000.0]))
    anomalies = estimator.update(batch)
    assert anomalies.shape == batch.shape
    assert anomalies[-1]
    assert estimator.threshold > 0
    assert estimator._history.size <= 10000
    assert np.isfinite(original_threshold)


def test_pot_masked_nan_bootstrap_and_validation():
    rng = np.random.default_rng(21)
    initial = rng.pareto(a=3.0, size=500)
    estimator = POTThreshold(tail_quantile=0.9, target_rate=0.02, min_exceedances=5)
    observed = estimator.update(initial)
    assert not observed.any()
    values = np.array([0.2, np.nan, 100.0])
    valid = np.array([True, True, False])
    detected = estimator.detect(values, mask=valid)
    assert not detected.any()
    assert np.isfinite(estimator.threshold)
    with pytest.raises(RuntimeError):
        POTThreshold().detect(np.ones(2))
    with pytest.raises(ValueError):
        estimator.fit(np.ones(10))
    with pytest.raises(ValueError):
        POTThreshold(target_rate=0.2)


def test_robust_detector_validation_and_one_dimensional_series():
    with pytest.raises(ValueError):
        RobustMADDetector(threshold=0)
    with pytest.raises(ValueError):
        RobustMADDetector(huber_delta=0)
    with pytest.raises(ValueError):
        RobustMADDetector(top_k=0)
    with pytest.raises(ValueError):
        RobustMADDetector(block_size=3)
    detector = RobustMADDetector(threshold=5.0)
    with pytest.raises(TypeError):
        detector.score(np.ones((3, 2)), axis=True)
    with pytest.raises(ValueError):
        detector.score(np.ones((3, 2)), axis=2)
    with pytest.raises(ValueError):
        detector.score(np.empty((0, 2)))
    series = np.array([1.0, 1.0, 1.0, 1.0, 20.0])
    scores = detector.score(series)
    assert scores.shape == series.shape
    assert detector.detect(series)[-1]


def test_pot_invalid_inputs_history_cap_and_rejected_update(monkeypatch):
    with pytest.raises(ValueError):
        POTThreshold(tail_quantile=1.0)
    with pytest.raises(ValueError):
        POTThreshold(target_rate=0.2)
    with pytest.raises(ValueError):
        POTThreshold(min_exceedances=2)
    with pytest.raises(ValueError):
        POTThreshold(history_size=0)
    estimator = POTThreshold(tail_quantile=0.9, target_rate=0.01, min_exceedances=5, history_size=100)
    with pytest.raises(RuntimeError):
        _ = estimator.threshold
    with pytest.raises(RuntimeError):
        estimator.detect(np.ones(3))
    rng = np.random.default_rng(33)
    calibration = rng.pareto(a=2.0, size=1000)
    with pytest.raises(ValueError):
        estimator.fit(calibration, mask=np.ones(10))
    with pytest.raises(ValueError):
        estimator.fit(np.full(20, np.nan), mask=np.zeros(20, dtype=bool))
    with pytest.raises(ValueError):
        estimator.fit(-calibration)
    estimator.fit(calibration)
    assert estimator._history.size == 100
    old_threshold = estimator.threshold
    assert estimator.update(np.array([1e6]))[0]
    assert estimator.threshold == old_threshold
    with pytest.raises(ValueError):
        estimator.update(np.array([-1.0]))


def test_pot_rejects_invalid_gpd_parameters(monkeypatch):
    import src.pot_threshold as pot_module

    rng = np.random.default_rng(45)
    calibration = rng.pareto(a=2.5, size=1000)
    estimator = POTThreshold(tail_quantile=0.9, target_rate=0.01, min_exceedances=5)
    monkeypatch.setattr(pot_module.genpareto, "fit", lambda *args, **kwargs: (np.nan, 0.0, 1.0))
    with pytest.raises(ValueError, match="invalid parameters"):
        estimator.fit(calibration)
