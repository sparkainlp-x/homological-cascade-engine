"""Tests for the experimental topological optics telemetry module."""

import numpy as np
import pytest

from src.analog_calibration import AnalogSoundCalibrator
from src.deep_mechanics.nonstationary_evt import NonStationaryPOTEstimator
from src.deep_mechanics.topological_laplacian import HigherDimensionalLaplacian
from src.homological_cascade import HomologicalMembraneChain
from src.quantum.topological_ao import TopologicalAdaptiveOpticsFilter


def test_topological_ao_reconstruction_and_attack_detection():
    rng = np.random.default_rng(seed=42)
    n_samples, n_channels = 500, 256

    # Simulate slowly varying common-mode atmospheric drift and sensor noise.
    time = np.linspace(0, 5, n_samples)
    kolmogorov_noise = 2.0 * np.sin(0.4 * time)[:, None] + rng.normal(
        0, 0.5, size=(n_samples, n_channels)
    )

    ao_filter = TopologicalAdaptiveOpticsFilter(n_subapertures=n_channels)
    ao_filter.calibrate_ambient_sky(kolmogorov_noise[:200])

    false_alarms = 0
    for frame in kolmogorov_noise[200:400]:
        is_attack, clean_phase, _ = ao_filter.process_photon_frame(frame)
        false_alarms += int(is_attack)
        assert np.var(clean_phase) <= np.var(frame) + 1e-5

    attack_frame = kolmogorov_noise[450].copy()
    attack_frame[50:100] += 18.0
    is_attack_detected, _, energy = ao_filter.process_photon_frame(attack_frame)

    assert false_alarms <= 2
    assert is_attack_detected is True
    assert energy > ao_filter.pot_estimator.threshold


def test_topological_filter_validation_and_calibration_state():
    with pytest.raises(TypeError):
        TopologicalAdaptiveOpticsFilter(n_subapertures=160.5)
    with pytest.raises(ValueError):
        TopologicalAdaptiveOpticsFilter(n_subapertures=159)
    with pytest.raises(ValueError):
        TopologicalAdaptiveOpticsFilter(n_subapertures=160, coupling_j=-1)
    with pytest.raises(ValueError):
        TopologicalAdaptiveOpticsFilter(n_subapertures=160, sample_rate_hz=0)

    ao_filter = TopologicalAdaptiveOpticsFilter(n_subapertures=160)
    with pytest.raises(RuntimeError, match="calibrated"):
        ao_filter.process_photon_frame(np.zeros(160))
    with pytest.raises(ValueError, match="shape"):
        ao_filter.calibrate_ambient_sky(np.zeros((50, 159)))
    with pytest.raises(ValueError, match="exceedances"):
        ao_filter.calibrate_ambient_sky(np.random.default_rng(7).normal(size=(20, 160)))
    assert not ao_filter.is_calibrated


def test_higher_dimensional_laplacian_hodge_components_and_validation():
    chain = HomologicalMembraneChain(n_0cells=3, n_1cells=8, n_2cells=2)
    laplacian = HigherDimensionalLaplacian(chain.d1, chain.d2)
    rng = np.random.default_rng(14)
    vector = rng.normal(size=8)
    exact, coexact, harmonic = laplacian.hodge_decomposition(vector)
    np.testing.assert_allclose(exact + coexact + harmonic, vector, atol=1e-12)
    assert np.linalg.norm(laplacian.d1 @ coexact) < 1e-10
    assert np.linalg.norm(laplacian.d2.T @ exact) < 1e-10
    assert laplacian.L_k.shape == (8, 8)
    assert np.linalg.eigvalsh(laplacian.L_k).min() > -1e-10

    empty_cells = HigherDimensionalLaplacian(np.zeros((2, 3)), np.zeros((3, 0)))
    _, _, residual = empty_cells.hodge_decomposition(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(residual, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="two-dimensional"):
        HigherDimensionalLaplacian(np.ones(3), np.ones((3, 1)))
    with pytest.raises(ValueError, match="dimensions"):
        HigherDimensionalLaplacian(np.ones((2, 3)), np.ones((4, 1)))
    with pytest.raises(ValueError, match="finite"):
        HigherDimensionalLaplacian(np.array([[np.nan]]), np.zeros((1, 0)))
    with pytest.raises(ValueError, match="satisfy d1"):
        HigherDimensionalLaplacian(np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="shape"):
        laplacian.hodge_decomposition(np.ones(7))
    with pytest.raises(ValueError, match="finite"):
        laplacian.hodge_decomposition(np.full(8, np.nan))


def test_analog_calibration_whitening_missing_values_and_validation():
    calibrator = AnalogSoundCalibrator(sample_rate_hz=1000, n_channels=3)
    with pytest.raises(RuntimeError, match="first"):
        calibrator.process_frame(np.ones(3))
    with pytest.raises(ValueError):
        AnalogSoundCalibrator(sample_rate_hz=np.nan, n_channels=3)
    with pytest.raises(ValueError):
        AnalogSoundCalibrator(sample_rate_hz=1000, n_channels=True)
    with pytest.raises(ValueError, match="shape"):
        calibrator.calibrate_baseline(np.ones(10))
    with pytest.raises(ValueError, match="two calibration"):
        calibrator.calibrate_baseline(np.ones((1, 3)))
    with pytest.raises(ValueError, match="two finite"):
        calibrator.calibrate_baseline(np.array([[np.nan, 1, 1], [np.nan, 2, 2]]))

    block = np.array([[1.0, 2.0, 5.0], [1.0, 4.0, 5.0], [1.0, 3.0, 5.0]])
    calibrator.calibrate_baseline(block)
    assert calibrator.is_calibrated
    clean = calibrator.process_frame(np.array([1.0, np.nan, 5.0]))
    assert np.isfinite(clean).all()
    assert clean[0] == 0.0
    assert clean[2] == 0.0
    assert calibrator.process_block_whitened(block).shape == block.shape
    assert calibrator.restore_scale(np.ones(3)).shape == (3,)
    with pytest.raises(ValueError, match="shape"):
        calibrator.process_frame(np.ones(2))
    with pytest.raises(ValueError, match="shape"):
        calibrator.process_block_whitened(np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        calibrator.restore_scale(np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        calibrator.restore_scale(np.array([np.nan, 1, 1]))


def test_nonstationary_evt_fit_streaming_and_rejections(monkeypatch):
    rng = np.random.default_rng(22)
    sample = rng.exponential(scale=2.0, size=1000)
    estimator = NonStationaryPOTEstimator(
        risk_level=0.001,
        init_quantile=0.9,
        min_exceedances=5,
        update_interval=1,
        history_size=200,
    )
    with pytest.raises(RuntimeError):
        estimator.update_dynamic_threshold(1.0)
    estimator.fit_initial_window(sample)
    old_threshold = estimator.threshold
    old_history_size = estimator._history.size
    detected, status = estimator.update_dynamic_threshold(1e6)
    assert detected
    assert estimator._history.size == old_history_size
    assert status["threshold"] == old_threshold
    detected, status = estimator.update_dynamic_threshold(float(np.median(sample)))
    assert not detected
    assert estimator._history.size == old_history_size
    assert estimator._history[-1] == pytest.approx(float(np.median(sample)))
    assert status["sigma"] > 0
    assert estimator.threshold > 0
    assert estimator._history.size <= 200

    with pytest.raises(ValueError):
        NonStationaryPOTEstimator(risk_level=0.2)
    with pytest.raises(ValueError):
        NonStationaryPOTEstimator(init_quantile=1.0)
    with pytest.raises(ValueError):
        NonStationaryPOTEstimator(min_exceedances=2)
    with pytest.raises(ValueError):
        NonStationaryPOTEstimator(history_size=0)
    with pytest.raises(ValueError):
        NonStationaryPOTEstimator(update_interval=0)
    with pytest.raises(ValueError):
        NonStationaryPOTEstimator(adaptation_rate=0)
    with pytest.raises(ValueError, match="finite"):
        NonStationaryPOTEstimator().fit_initial_window(np.array([np.nan, np.inf]))
    with pytest.raises(ValueError, match="non-negative"):
        NonStationaryPOTEstimator().fit_initial_window(np.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="exceedances"):
        NonStationaryPOTEstimator().fit_initial_window(np.arange(10.0))
    with pytest.raises(ValueError, match="scalar"):
        estimator.update_dynamic_threshold(np.ones(2))
    with pytest.raises(ValueError, match="finite and non-negative"):
        estimator.update_dynamic_threshold(-1.0)


def test_nonstationary_evt_rejects_invalid_fitted_parameters(monkeypatch):
    import src.deep_mechanics.nonstationary_evt as evt_module

    rng = np.random.default_rng(23)
    values = rng.exponential(size=400)
    estimator = NonStationaryPOTEstimator(risk_level=0.001, init_quantile=0.9)
    monkeypatch.setattr(evt_module.genpareto, "fit", lambda *args, **kwargs: (np.nan, 0.0, 1.0))
    with pytest.raises(ValueError, match="invalid parameters"):
        estimator.fit_initial_window(values)
