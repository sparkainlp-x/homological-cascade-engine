"""Verification for topological adaptive-optics filtering and QKD triage."""

import numpy as np
import pytest

from src.analog_calibration import AnalogSoundCalibrator
from src.quantum.topological_ao import TopologicalAdaptiveOpticsFilter


def test_topological_ao_reconstruction_and_attack_detection():
    rng = np.random.default_rng(seed=42)
    n_samples, n_channels = 500, 256

    # Simulate smooth atmospheric drift with independent detector noise.
    time = np.linspace(0, 5, n_samples)
    kolmogorov_noise = (
        2.0 * np.sin(0.4 * time)[:, None]
        + rng.normal(0, 0.5, size=(n_samples, n_channels))
    )

    ao_filter = TopologicalAdaptiveOpticsFilter(n_subapertures=n_channels)
    ao_filter.calibrate_ambient_sky(kolmogorov_noise[:200])
    assert ao_filter.is_calibrated

    false_alarms = 0
    for i in range(200, 400):
        frame = kolmogorov_noise[i]
        is_attack, clean_phase, _ = ao_filter.process_photon_frame(frame)
        false_alarms += int(is_attack)
        assert clean_phase.shape == (n_channels,)
        assert np.var(clean_phase) <= np.var(frame) + 1e-5

    # A localized phase discontinuity models a deliberately injected anomaly.
    attack_frame = kolmogorov_noise[450].copy()
    attack_frame[50:100] += 18.0
    is_attack_detected, reconstructed, energy = ao_filter.process_photon_frame(
        attack_frame
    )

    assert false_alarms <= 2
    assert is_attack_detected is True
    assert reconstructed.shape == attack_frame.shape
    assert energy > 0.0


def test_filter_requires_calibration_and_rejects_invalid_frames():
    filter_ = TopologicalAdaptiveOpticsFilter(n_subapertures=8)
    with pytest.raises(RuntimeError, match="calibrated"):
        filter_.process_photon_frame(np.zeros(8))

    filter_.calibrate_ambient_sky(np.zeros((12, 8)))
    with pytest.raises(ValueError, match="shape"):
        filter_.process_photon_frame(np.zeros(7))
    with pytest.raises(ValueError, match="finite"):
        filter_.process_photon_frame(np.full(8, np.nan))


def test_filter_parameter_validation():
    with pytest.raises(TypeError, match="n_subapertures"):
        TopologicalAdaptiveOpticsFilter(n_subapertures=2.5)
    with pytest.raises(ValueError, match="n_subapertures"):
        TopologicalAdaptiveOpticsFilter(n_subapertures=0)
    with pytest.raises(ValueError, match="coupling_j"):
        TopologicalAdaptiveOpticsFilter(n_subapertures=4, coupling_j=-1.0)
    with pytest.raises(ValueError, match="coupling_j"):
        TopologicalAdaptiveOpticsFilter(n_subapertures=4, coupling_j=np.inf)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        TopologicalAdaptiveOpticsFilter(n_subapertures=4, sample_rate_hz=0)


def test_analog_calibrator_offsets_relative_gains_and_validation():
    rng = np.random.default_rng(7)
    baseline = np.column_stack(
        [4.0 + rng.normal(0, 0.25, 300), -2.0 + rng.normal(0, 0.5, 300)]
    )
    calibrator = AnalogSoundCalibrator(sample_rate_hz=5_000, n_channels=2)
    with pytest.raises(RuntimeError, match="calibrate_baseline"):
        calibrator.process_frame(np.zeros(2))

    calibrator.calibrate_baseline(baseline)
    assert calibrator.is_calibrated
    np.testing.assert_allclose(calibrator.baseline_offset, [4.0, -2.0], atol=0.1)
    assert calibrator.gain[0] > calibrator.gain[1]
    processed = calibrator.process_block_whitened(baseline)
    assert processed.shape == baseline.shape
    assert np.all(np.isfinite(processed))
    np.testing.assert_allclose(np.median(processed, axis=0), 0.0, atol=0.1)
    np.testing.assert_allclose(calibrator.process_frame(baseline[0]), processed[0])
    assert calibrator.process_block_whitened(baseline[0]).shape == (2,)

    with pytest.raises(ValueError, match="shape"):
        calibrator.calibrate_baseline(np.ones((4, 3)))
    with pytest.raises(ValueError, match="finite"):
        calibrator.calibrate_baseline(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="positive"):
        AnalogSoundCalibrator(n_channels=0)
    with pytest.raises(TypeError, match="n_channels"):
        AnalogSoundCalibrator(n_channels=1.5)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        AnalogSoundCalibrator(sample_rate_hz=-1, n_channels=1)
